"""把 A_Memorix 的旧记忆迁进 memfade。

为什么要迁：A_Memorix 存了几十条关于人的事实，但里面**近似重复很严重**
（实测：「称对方为宝宝」3 条、「二十块加价与铁子无关」4 条，
 换着说法反复记）。memfade 的去重正好治这个。

迁移脚本做三件事：

1. 只取**有价值的事实**（fact_claims 的 stable_fact + person_fact 段落），
   不搬聊天摘要；
2. 用 memfade 自己的去重逻辑合并近似重复；
3. 默认落到**私聊作用域**——事实是关于「这个人」的，
   不是关于某个群的；跨会话共享要显式开（``--cross-scope``）。

用法：

    python scripts/import_a_memorix.py \
        --source "G:/AI/许杏玖-MaiBot/MaiBot/data/a-memorix/metadata/metadata.db" \
        --target "G:/AI/许杏玖-MaiBot/MaiBot/data/plugins/memfade.memory/memfade.db" \
        --scope qq:private:1738157278 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade import MemoryDraft, MemoryEngine, MemoryKind, MemFadeConfig, Scope  # noqa: E402
from memfade.textutil import normalized_hash, similarity  # noqa: E402


def read_source(db_path: str) -> List[Dict[str, Any]]:
    """读 A_Memorix 里值得迁移的事实。"""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: List[Dict[str, Any]] = []
    try:
        try:
            rows = conn.execute(
                "SELECT value_text, polarity, profile_section, scope_id FROM fact_claims"
            ).fetchall()
            for row in rows:
                text = str(row["value_text"] or "").strip()
                if text:
                    out.append({"content": text, "origin": "fact_claim", "scope_id": row["scope_id"]})
        except sqlite3.OperationalError:
            pass
        try:
            rows = conn.execute(
                "SELECT content, source FROM paragraphs WHERE source LIKE 'person_fact:%'"
            ).fetchall()
            for row in rows:
                text = str(row["content"] or "").strip()
                if text:
                    out.append({"content": text, "origin": "paragraph", "scope_id": row["source"]})
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()
    return out


def load_llm_from_maibot(config_path: str) -> Tuple[str, str, str]:
    """从 MaiBot 自己的 model_config.toml 里取 LLM 端点。

    这样用户不用手抄密钥——迁移脚本和宿主读同一份配置。
    返回 (base_url, api_key, model)。
    """

    import tomllib

    with open(config_path, "rb") as handle:
        config = tomllib.load(handle)
    providers = config.get("api_providers") or []
    if isinstance(providers, dict):
        providers = list(providers.values())
    tasks = config.get("model_task_config") or {}
    model_names: List[str] = []
    for task in ("memory", "replyer", "utils"):
        model_list = (tasks.get(task) or {}).get("model_list") or []
        model_names.extend(str(x) for x in model_list)
    models = config.get("models") or []
    for entry in models:
        if str(entry.get("name")) in model_names:
            provider_name = entry.get("api_provider")
            for provider in providers:
                if provider.get("name") == provider_name or provider.get("model_identifier") == provider_name:
                    return (
                        str(provider.get("base_url") or ""),
                        str(provider.get("api_key") or ""),
                        str(entry.get("model_identifier") or entry.get("name") or ""),
                    )
    raise SystemExit(f"{config_path} 里找不到可用的 LLM 端点")


def canonicalize_via_plugin(
    texts: List[str],
    webui: str,
    token: str,
    chunk_size: int = 20,
) -> List[Dict[str, Any]]:
    """让**运行中的插件**做规范化（它用的是宿主自己的 LLM 服务）。

    ⚠ 不要用裸 HTTP 直连模型端点：MiniMax 这类模型默认开着 thinking，
    裸调用容易把 token 全烧在思考上、最后输出为空
    （实测 finish_reason=length、content 空、reasoning 两万 token）。
    宿主自己的 LLM 服务把这些参数管好了。

    ⚠ **必须批量问**：一条一条问的话，模型看不到彼此，
    就没有机会把「同件事的多种说法」合并成一条——
    而那恰恰是旧库里最严重的问题。
    """

    import json as _json
    import urllib.request

    out: List[Dict[str, Any]] = []
    url = webui.rstrip("/") + "/api/webui/plugins/runtime/webui/memfade.memory/memfade_overview/actions/canonicalize"
    headers = {"Content-Type": "application/json"}
    # 宿主的反爬中间件会按 User-Agent 拦请求（curl 的 UA 会被当成爬虫），
    # 带 Cookie 的已认证请求本该放行，但不同版本的放行策略不一致，
    # 这里老实给一个正常 UA，省得被 403。
    headers["User-Agent"] = "memfade-importer/0.1"
    headers["Accept"] = "application/json"
    if token:
        headers["Cookie"] = f"maibot_session={token}"
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        # 列表走 JSON 字符串传：WebUI 的声明式 action 只允许标量参数
        body = _json.dumps({"args": {"facts": _json.dumps(chunk, ensure_ascii=False)}}).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = _json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"  ! 规范化第 {start // chunk_size + 1} 批失败：{exc}")
            for text in chunk:
                out.append({"content": text, "subject": "", "predicate": "", "object": ""})
            continue
        result = payload.get("result") or {}
        if not result.get("ok"):
            print(f"  ! 规范化第 {start // chunk_size + 1} 批失败：{result.get('error')}")
            for text in chunk:
                out.append({"content": text, "subject": "", "predicate": "", "object": ""})
            continue
        out.extend(result.get("items") or [])
    return out


def canonicalize_via_host(
    texts: List[str],
    host_root: str,
    chunk_size: int = 4,
) -> List[Dict[str, Any]]:
    """直接调用宿主自己的 LLM 服务做规范化（推荐路径）。

    比走 WebUI 好在两点：
    1. **没有请求超时**——WebUI 那条路 41 条事实会 504；
    2. 参数由宿主统一管理（MiniMax 这类默认开 thinking 的模型，
       宿主会处理好 token 预算）。

    需要在 MaiBot 自己的 venv 下跑（要能 import src.*）。
    """

    import asyncio
    import json as _json

    if host_root and host_root not in sys.path:
        sys.path.insert(0, host_root)
    from src.services import llm_service  # type: ignore

    from memfade.extract import CANONICALIZE_PROMPT, parse_json_block

    async def _run() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            out.extend(await _run_chunk(chunk))
        return out

    async def _run_chunk(chunk: List[str]) -> List[Dict[str, Any]]:
        try:
            return await _one_batch(chunk)
        except Exception as exc:
            # 失败就**拆成两半**再试：模型在长输入下容易思考超时。
            if len(chunk) > 1:
                mid = len(chunk) // 2
                return (await _run_chunk(chunk[:mid])) + (await _run_chunk(chunk[mid:]))
            print(f"  ! 规范化失败（{len(chunk)} 条）：{exc}")
            return [{"content": t, "subject": "", "predicate": "", "object": ""} for t in chunk]

    async def _one_batch(chunk: List[str]) -> List[Dict[str, Any]]:
        prompt = CANONICALIZE_PROMPT.format(facts="\n".join(f"- {t}" for t in chunk))
        request = llm_service.LLMServiceRequest(
            task_name="memory",
            request_type="memfade.canonicalize",
            prompt=prompt,
            temperature=0.1,
            # MiniMax 这类模型默认开着 thinking，会把预算全烧在思考上：
            # max_tokens 太小 -> finish_reason=length、输出为空；
            # 太大 -> 超过宿主的 120 秒超时。
            # 实测 4 条一批 + 8000 token 最稳。
            max_tokens=8000,
        )
        result = await llm_service.generate(request)
        payload = result.to_capability_payload()
        raw = str(payload.get("response") or "")
        items = parse_json_block(raw)
        if not isinstance(items, list) or not items:
            raise RuntimeError("模型没有返回可解析的 JSON")
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            out.append(
                {
                    "content": content,
                    "subject": str(item.get("subject") or "").strip(),
                    "predicate": str(item.get("predicate") or "").strip(),
                    "object": str(item.get("object") or "").strip(),
                }
            )
        if not out:
            raise RuntimeError("模型返回了空结果")
        return out

    return asyncio.run(_run())


def split_via_host(
    texts: List[str],
    host_root: str,
    chunk_size: int = 2,
) -> List[List[Dict[str, Any]]]:
    """把每条长记忆拆成多条原子事实。

    返回与 ``texts`` **一一对应**：第 i 项是第 i 条输入拆出来的事实列表。
    空列表表示这条没拆成功——调用方必须**保留原条目**。

    为什么不能像 canonicalize_via_host 那样「失败就返回原文」：
    那等于把一条 284 字的长记忆删掉、又塞回一条 284 字的长记忆，
    还顺手丢了三元组——记忆没变好，反而被改坏了。
    「拆不动就别动」才是对的。

    与 canonicalize_via_host 的另一个区别：输出按**输入编号**组织
    （JSON 对象的键就是编号），所以拆得对不对能逐条核对，
    不会出现「模型把 3 条输入合成 1 条」时对不上号。
    """

    import asyncio

    if host_root and host_root not in sys.path:
        sys.path.insert(0, host_root)
    from src.services import llm_service  # type: ignore

    from memfade.extract import CANONICALIZE_SPLIT_PROMPT, parse_json_block

    async def _run() -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            out.extend(await _run_chunk(chunk))
        return out

    async def _run_chunk(chunk: List[str]) -> List[List[Dict[str, Any]]]:
        try:
            return await _one_batch(chunk)
        except Exception as exc:
            # 失败就**拆成两半**再试：模型在长输入下容易思考超时。
            if len(chunk) > 1:
                mid = len(chunk) // 2
                return (await _run_chunk(chunk[:mid])) + (await _run_chunk(chunk[mid:]))
            print(f"  ! 拆解失败（{len(chunk)} 条）：{exc}")
            return [[] for _ in chunk]

    async def _one_batch(chunk: List[str]) -> List[List[Dict[str, Any]]]:
        prompt = CANONICALIZE_SPLIT_PROMPT.format(facts="\n".join(f"- {t}" for t in chunk))
        request = llm_service.LLMServiceRequest(
            task_name="memory",
            request_type="memfade.split",
            prompt=prompt,
            temperature=0.1,
            max_tokens=8000,
        )
        result = await llm_service.generate(request)
        payload = result.to_capability_payload()
        raw = str(payload.get("response") or "")
        items = parse_json_block(raw)
        if not isinstance(items, dict) or not items:
            raise RuntimeError("模型没有返回可解析的 JSON 对象")

        out: List[List[Dict[str, Any]]] = []
        for index in range(len(chunk)):
            raw_items = items.get(str(index))
            if not isinstance(raw_items, list):
                # 模型漏了这一条 → 保留原条目，不要瞎猜。
                print(f"  ! 模型没给出第 {index} 条的拆解结果，保留原条目")
                out.append([])
                continue
            facts: List[Dict[str, Any]] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                facts.append(
                    {
                        "content": content,
                        "subject": str(item.get("subject") or "").strip(),
                        "predicate": str(item.get("predicate") or "").strip(),
                        "object": str(item.get("object") or "").strip(),
                    }
                )
            out.append(facts)
        return out

    return asyncio.run(_run())


def fill_triples_via_host(
    texts: List[str],
    host_root: str,
    chunk_size: int = 8,
    max_tokens: int = 4000,
) -> List[Dict[str, str]]:
    """给一批已经拆好的事实补三元组。

    与 ``split_via_host`` 分开是有意的：

    * **拆**是长文本 → 多条事实，输出大、模型要想很久，容易超时；
    * **补三元组**是一条事实 → 三个短字段，输出小，13 秒就完事。

    所以正确的顺序是：先用本地规则拆（确定、免费、不丢字），
    再让模型给拆好的碎片补三元组。

    返回与 ``texts`` 等长的列表，每项是
    ``{"subject": ..., "predicate": ..., "object": ...}``；
    模型没给的就留空字符串——空三元组不影响检索，
    矛盾检测还有内容兜底那条路。
    """

    import asyncio

    if host_root and host_root not in sys.path:
        sys.path.insert(0, host_root)
    from src.services import llm_service  # type: ignore

    from memfade.extract import TRIPLE_PROMPT, parse_json_block

    empty = {"subject": "", "predicate": "", "object": ""}

    async def _run() -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            out.extend(await _run_chunk(chunk))
        return out

    async def _run_chunk(chunk: List[str]) -> List[Dict[str, str]]:
        try:
            return await _one_batch(chunk)
        except Exception as exc:
            if len(chunk) > 1:
                mid = len(chunk) // 2
                return (await _run_chunk(chunk[:mid])) + (await _run_chunk(chunk[mid:]))
            print(f"  ! 补三元组失败（{len(chunk)} 条）：{exc}")
            return [dict(empty) for _ in chunk]

    async def _one_batch(chunk: List[str]) -> List[Dict[str, str]]:
        prompt = TRIPLE_PROMPT.format(facts="\n".join(f"- {t}" for t in chunk))
        request = llm_service.LLMServiceRequest(
            task_name="memory",
            request_type="memfade.triples",
            prompt=prompt,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        result = await llm_service.generate(request)
        payload = result.to_capability_payload()
        raw = str(payload.get("response") or "")
        items = parse_json_block(raw)
        if not isinstance(items, list) or not items:
            raise RuntimeError("模型没有返回可解析的 JSON 数组")

        by_index: Dict[int, Dict[str, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            by_index[index] = {
                "subject": str(item.get("subject") or "").strip(),
                "predicate": str(item.get("predicate") or "").strip(),
                "object": str(item.get("object") or "").strip(),
            }
        return [by_index.get(i, dict(empty)) for i in range(len(chunk))]

    return asyncio.run(_run())


def read_webui_token(config_path: str) -> str:
    """从 MaiBot 的 data/webui.json 里读 WebUI 令牌。"""

    import json as _json

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return str(_json.load(handle).get("access_token") or "")
    except Exception:
        return ""


def mai_session_id(platform: str, user_id: str, group_id: str, account_id: str, scope: str) -> str:
    """按 MaiBot 的算法算 session_id。

    MaiBot 的 ``SessionUtils.calculate_session_id``：

    - 群聊：``md5(platform + account + scope + group_id)``
    - 私聊：``md5(platform + account + scope + user_id + "private")``

    memfade 的作用域键就是它——所以导入时要算准，
    否则记忆会落到一个机器人根本读不到的会话里。
    """

    import hashlib

    parts = [platform]
    if account_id:
        parts.append(f"account:{account_id}")
    if scope:
        parts.append(f"scope:{scope}")
    if group_id:
        parts.append(group_id)
    else:
        parts.extend([user_id, "private"])
    return hashlib.md5("_".join(parts).encode()).hexdigest()

def classify(text: str) -> Tuple[MemoryKind, float]:
    """给旧事实估个种类与显著性。"""

    if any(word in text for word in ("喜欢", "爱好", "讨厌", "不吃", "最爱")):
        return MemoryKind.PREFERENCE, 0.75
    if any(word in text for word in ("称为", "称呼", "叫作", "铁子", "朋友")):
        return MemoryKind.RELATION, 0.75
    if any(word in text for word in ("可以", "无法", "会", "不能", "已经", "正在")):
        return MemoryKind.SEMANTIC, 0.7
    return MemoryKind.SEMANTIC, 0.65


def main() -> int:
    parser = argparse.ArgumentParser(description="把 A_Memorix 的旧事实迁进 memfade")
    parser.add_argument("--source", required=True, help="A_Memorix 的 metadata.db")
    parser.add_argument("--target", required=True, help="memfade 的 memfade.db")
    parser.add_argument("--scope", default="", help="目标作用域，如 qq:private:1738157278")
    parser.add_argument("--cross-scope", action="store_true", help="允许跨会话共享（默认否）")
    parser.add_argument("--dry-run", action="store_true", help="只看会迁什么，不写库")
    parser.add_argument("--limit", type=int, default=0, help="最多迁多少条（0=不限）")
    parser.add_argument(
        "--plugin-webui",
        default="",
        help="MaiBot WebUI 地址（如 http://127.0.0.1:8001）；给了就用运行中的插件规范化旧事实",
    )
    parser.add_argument(
        "--host-root",
        default="",
        help="MaiBot 源码根目录；给了就直接调宿主的 LLM 服务（推荐，没有超时问题）",
    )
    parser.add_argument(
        "--webui-config",
        default="",
        help="MaiBot 的 data/webui.json；用来读 WebUI 令牌",
    )
    parser.add_argument(
        "--token",
        default="",
        help="WebUI 令牌（不给就按 --webui-config 读）",
    )
    parser.add_argument(
        "--cache",
        default="",
        help="规范化结果的缓存文件（复用它就不必重跑 LLM）",
    )
    parser.add_argument("--maibot-account", default="", help="MaiBot 账号 ID（算 session_id 用）")
    parser.add_argument("--maibot-user", default="", help="私聊对方的 QQ 号（算 session_id 用）")
    parser.add_argument("--maibot-group", default="", help="群号（算 session_id 用）")
    parser.add_argument("--maibot-scope", default="", help="路由作用域（算 session_id 用）")
    parser.add_argument(
        "--self-names",
        default="",
        help="机器人自己的名字（逗号分隔）：主语是它的事实不迁",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="只迁这些主语的事（逗号分隔，空=不限制）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"找不到源库：{args.source}", file=sys.stderr)
        return 1
    scope_key = args.scope
    if not scope_key and (args.maibot_user or args.maibot_group):
        session = mai_session_id(
            "qq", args.maibot_user, args.maibot_group, args.maibot_account, args.maibot_scope
        )
        # ⚠ chat_type 必须是 "session"——MaiBot 适配器用的就是这个，
        # 写成 group/private 会导致机器人根本读不到导入的记忆。
        scope_key = f"qq:session:{session}"
    scope_key = scope_key or "qq:private:imported"
    parts = scope_key.split(":", 2)
    scope = Scope(
        platform=parts[0] if parts else "qq",
        chat_type=parts[1] if len(parts) > 1 else "private",
        chat_id=parts[2] if len(parts) > 2 else "imported",
        cross_scope=bool(args.cross_scope),
    )

    source_rows = read_source(args.source)
    if args.limit:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        print("源库里没有可迁移的事实")
        return 0

    # 先看看近似重复有多少
    hashes = Counter(normalized_hash(r["content"]) for r in source_rows)
    exact_dupes = sum(n - 1 for n in hashes.values() if n > 1)
    near = 0
    for i in range(len(source_rows)):
        for j in range(i + 1, len(source_rows)):
            a, b = source_rows[i]["content"], source_rows[j]["content"]
            if normalized_hash(a) != normalized_hash(b) and similarity(a, b) >= 0.82:
                near += 1

    print("=" * 66)
    print("A_Memorix -> memfade 迁移")
    print("=" * 66)
    print(f"  源库可迁移事实 : {len(source_rows)} 条")
    print(f"  完全重复       : {exact_dupes} 条")
    print(f"  近似重复对     : {near} 对（阈值 0.82）")
    print(f"  目标作用域     : {scope.key()}（跨会话共享={scope.cross_scope}）")
    if args.dry_run:
        print("  --dry-run：没有写库")
        return 0

    if args.host_root:
        print("  用宿主 LLM 服务规范化旧事实…")
        raw_texts = [r["content"] for r in source_rows]
        cache_path = args.cache or os.path.join(os.path.dirname(args.target) or ".", "memfade_canonical_cache.json")
        canonical = None
        if args.cache and os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    canonical = json.load(handle)
                print(f"  用缓存的规范化结果（{len(canonical)} 条）")
            except Exception:
                canonical = None
        if canonical is None:
            canonical = canonicalize_via_host(raw_texts, args.host_root)
            try:
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(canonical, handle, ensure_ascii=False, indent=1)
                print(f"  规范化结果已缓存到 {cache_path}")
            except Exception:
                pass
        by_text = {r["content"]: r for r in source_rows}
        # 规范化之后**只留规范版**：原文已经被它覆盖了，
        # 两份都存就等于没去重。
        source_rows = []
        for item in canonical:
            origin = by_text.get(item["content"], {})
            source_rows.append(
                {
                    "content": item["content"],
                    "origin": f"canonical:{origin.get('origin', 'fact_claim')}",
                    "scope_id": origin.get("scope_id", ""),
                    "subject": item.get("subject", ""),
                    "predicate": item.get("predicate", ""),
                    "object": item.get("object", ""),
                }
            )
        print(f"  规范化后共 {len(canonical)} 条")
    elif args.plugin_webui:
        token = args.token
        if not token and args.webui_config and os.path.isfile(args.webui_config):
            token = read_webui_token(args.webui_config)
        print(f"  用运行中的插件规范化旧事实（{args.plugin_webui}）…")
        raw_texts = [r["content"] for r in source_rows]
        canonical = canonicalize_via_plugin(raw_texts, args.plugin_webui, token)
        by_text = {r["content"]: r for r in source_rows}
        source_rows = []
        for item in canonical:
            origin = by_text.get(item["content"], {})
            source_rows.append(
                {
                    "content": item["content"],
                    "origin": f"canonical:{origin.get('origin', 'fact_claim')}",
                    "scope_id": origin.get("scope_id", ""),
                    "subject": item.get("subject", ""),
                    "predicate": item.get("predicate", ""),
                    "object": item.get("object", ""),
                }
            )
        print(f"  规范化后共 {len(canonical)} 条")

    config = MemFadeConfig.from_dict({"storage": {"db_path": args.target, "enable_vector": False}})
    engine = MemoryEngine(config)
    try:
        drafts = []
        for row in source_rows:
            kind, importance = classify(row["content"])
            drafts.append(
                MemoryDraft(
                    content=row["content"],
                    kind=kind,
                    importance=importance,
                    subject=row.get("subject", ""),
                    predicate=row.get("predicate", ""),
                    object=row.get("object", ""),
                    source=f"a_memorix:{row['origin']}",
                    cross_scope=scope.cross_scope,
                )
            )
        # 过一道「主语纪律」：迁移旧事实最容易把**机器人自己的事**
        # 一起搬过来（旧库里也有机器人的人设条目）。
        if drafts:
            from memfade.extract import filter_drafts

            kept, dropped = filter_drafts(
                drafts,
                self_names=tuple(n.strip() for n in (args.self_names or "").split(",") if n.strip()),
                focus_subjects=tuple(n.strip() for n in (args.focus or "").split(",") if n.strip()),
                participants=(),
                require_person=False,
                default_subject="",
            )
            for content, reason in dropped:
                print("  丢弃（%s）：%s" % (reason, content[:50]))
            drafts = kept
        before = engine.store.count(scope.key())
        written = engine.remember_many(drafts, scope)
        after = engine.store.count(scope.key())
        print(f"  实际入库       : {after - before} 条（读入 {len(drafts)} 条）")
        print(f"  合并/去重掉   : {len(drafts) - (after - before)} 条")
        print()
        print("  入库后的记忆：")
        for mem in engine.store.iter_scope(scope.key(), statuses=("active",), limit=40):
            print(f"    · [{mem.kind.value}] {mem.content[:60]}")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

