"""核对记忆里的事，聊天记录里到底有没有出现过。

## 为什么需要这个

记忆库被污染的方式不止一种：引用被当成用户的话、转发公告被当成用户的事、
**规范化时模型自己编**——最后这种最难发现，因为编出来的东西
读起来特别像真的。

真事：「季怡想在小红书上卖AI头像赚钱」。
编的：「季怡运营小红书账号，简介"AI炼丹师，炼到什么发什么喵"」
——「AI炼丹师」这四个字在 630 条聊天记录里**一次都没出现过**，
却是库里躺得好好的的一条记忆。

所以：**拿记忆回聊天记录里查一遍**。
每条记忆取它最有辨识度的几个词，看聊天记录里有没有；
没有的，要么是编的，要么是聊天记录已经滚出去了（后者要人工看）。

## 怎么用

    python scripts/verify_against_chat.py \
        --db "G:/AI/许杏玖-MaiBot/MaiBot/data/plugins/memfade.memory/memfade.db" \
        --messages "G:/AI/许杏玖-MaiBot/MaiBot/data/MaiBot.db"
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from typing import List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.textutil import bigrams, normalized_hash  # noqa: E402


_TERM_SPLIT = re.compile(r"[，,。；;！!？?、\s（）()【】\[\]「」『』“”\"'《》…—\-]+")
_STOP_TERMS = {
    "用户",
    "对方",
    "他",
    "她",
    "机器人",
    "已经",
    "正在",
    "可以",
    "不会",
    "没有",
    "自己",
    "什么",
    "一个",
    "这些",
    "那些",
    "不是",
    "就是",
    "现在",
    "当时",
    "以后",
    "以前",
}


def _terms(text: str) -> List[str]:
    """把一句话拆成可核对的词（按标点切，去掉虚词与过短的）。"""

    out: List[str] = []
    for chunk in _TERM_SPLIT.split(text or ""):
        chunk = (chunk or "").strip()
        if len(chunk) < 2 or chunk in _STOP_TERMS:
            continue
        out.append(chunk)
    return out


def load_chat(messages_db: str) -> str:
    """从 MaiBot 的消息库读出全部聊天正文。"""

    conn = sqlite3.connect("file:%s?mode=ro" % messages_db.replace("\\", "/"), uri=True)
    try:
        parts: List[str] = []
        for row in conn.execute(
            "SELECT processed_plain_text, raw_content FROM mai_messages "
            "WHERE processed_plain_text IS NOT NULL OR raw_content IS NOT NULL"
        ):
            for value in row:
                if value is None:
                    continue
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "replace")
                parts.append(str(value))
        return "\n".join(parts)
    finally:
        conn.close()


def load_messages(messages_db: str) -> List[str]:
    """读出聊天记录（一条一句，带时间与说话人）。"""

    conn = sqlite3.connect("file:%s?mode=ro" % messages_db.replace("\\", "/"), uri=True)
    try:
        out: List[str] = []
        for row in conn.execute(
            "SELECT timestamp, user_nickname, processed_plain_text, raw_content "
            "FROM mai_messages ORDER BY id"
        ):
            text = row[2] or row[3] or ""
            if isinstance(text, bytes):
                text = text.decode("utf-8", "replace")
            text = str(text).replace("\n", " ").strip()
            if not text:
                continue
            out.append("%s：%s" % (row[1] or "?", text))
        return out
    finally:
        conn.close()


def _relevant(messages: List[str], content: str, limit: int = 8) -> List[str]:
    """挑出与这条记忆最相关的聊天片段（按共享的词打分）。"""

    # 按**字符二元组**的重合度打分——记忆是改写过的，
    # 整词匹配会一条都找不到（跟核心库的检索一个道理）。
    query = set(bigrams(content or ""))
    if not query:
        return []
    scored = []
    for index, message in enumerate(messages):
        grams = set(bigrams(message))
        score = len(query & grams)
        if score >= 2:
            scored.append((score, index))
    scored.sort(reverse=True)
    picked = sorted({index for _, index in scored[:limit]})
    # 每条记忆再带上前后各一句，语境才看得出来
    out: List[str] = []
    for index in picked:
        for near in (index - 1, index, index + 1):
            if 0 <= near < len(messages) and messages[near] not in out:
                out.append(messages[near])
    return out


VERIFY_PROMPT = (
    "下面是几条聊天记录里抽出来的记忆，以及相关的聊天片段。\n"
    "请逐条判断：**这条记忆在聊天片段里有没有依据**。\n"
    "\n"
    "⚠ 聊天片段里，用户的消息**开头可能是机器人被引用的原话**"
    "（宿主把引用拼在正文前面）。判断依据时，"
    "**只算主人自己说的那部分**（引号后、最后那几句），"
    "引用的开头不算他说的。\n"
    "\n"
    "判据：\n"
    "- 聊天片段里用户（不是机器人）说过这件事 → 有依据；\n"
    "- 说话换了个说法、但说的是同一件事 → 有依据；\n"
    "- 细节（时间、集数、账号名、简介、金额）是聊天里从没出现过的 → 没有依据；\n"
    "- 聊天片段里只有机器人说过 → 没有依据"
    "（机器人的话不能当用户的依据）；\n"
    "- 聊天片段根本没提到 → 没有依据。\n"
    "\n"
    "只输出 JSON 数组，每项 {\"index\": 编号, \"verdict\": \"有依据|没依据\", \"note\": \"一句话\"}：\n"
    "{facts}"
)


async def _verify_with_llm(
    rows: List[dict],
    messages: List[str],
    host_root: str,
    batch_size: int = 4,
    llm_base_url: str = "",
    llm_model: str = "",
    llm_api_key: str = "",
) -> List[dict]:
    """用宿主 LLM 逐条判断记忆有没有依据。

    宿主 LLM 的来源有两条路：

    * ``host_root`` 给了且能 import 到 MaiBot 的 ``src.services.llm_service``
      —— 用它（MaiBot 部署的常规路径）；
    * 否则退到核心自带的 ``memfade.llm.HttpLLMClient``，用
      ``--llm-base-url / --llm-model / --llm-api-key`` 指一个 OpenAI 兼容接口。

    原来这里直接 ``from src.services import llm_service``，在非 MaiBot 宿主
    （比如 AstrBot）下必崩——工具应该跟宿主无关。
    """

    llm_client = None
    if host_root and host_root not in sys.path:
        sys.path.insert(0, host_root)
    try:
        from src.services import llm_service  # type: ignore

        use_host = True
    except Exception:
        use_host = False

    from memfade.extract import parse_json_block

    if not use_host:
        if not (llm_base_url and llm_model):
            return []
        from memfade.llm import HttpLLMClient

        llm_client = HttpLLMClient(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model,
        )

    verdicts: List[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        blocks = []
        for offset, row in enumerate(chunk):
            context = "\n".join(_relevant(messages, row["content"] or ""))
            blocks.append(
                "记忆 %d：%s\n相关片段：\n%s"
                % (offset, row["content"] or "", context or "（没找到相关片段）")
            )
        # ⚠ 提示词里有字面花括号（JSON 示例），不能用 .format()——
        # 「所有提示词都能被 format」是仓库里的一条守卫测试。
        prompt = VERIFY_PROMPT.replace("{facts}", "\n\n".join(blocks))
        if use_host:
            request = llm_service.LLMServiceRequest(
                task_name="memory",
                request_type="memfade.verify",
                prompt=prompt,
                temperature=0.1,
                # MiniMax 这类模型默认开 thinking，思考与正文共用预算——
                # 给小了正文会被截断（仓库规则 L5.6 记着这个坑）。
                max_tokens=6000,
            )
            result = await llm_service.generate(request)
            raw_text = str(result.to_capability_payload().get("response") or "")
        else:
            raw_text = await llm_client.complete(
                prompt,
                temperature=0.1,
                max_tokens=6000,
            )
        items = parse_json_block(raw_text)
        if not isinstance(items, list):
            print("  ! 这一批没解析出 JSON，跳过")
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(chunk):
                verdicts.append(
                    {
                        "id": chunk[index]["id"],
                        "content": chunk[index]["content"],
                        "verdict": str(item.get("verdict") or ""),
                        "note": str(item.get("note") or ""),
                    }
                )
    return verdicts


def _normalize(text: str) -> str:
    return normalized_hash(text).replace(" ", "")


def verify(content: str, chat: str) -> Tuple[bool, List[str]]:
    """一条记忆在聊天记录里找得到吗？返回（找得到, 可疑的词）。

    判据是**字面覆盖率**，不是整句匹配：
    记忆是改写过的（「买了」→「已经购买」），整句永远对不上；
    但**编出来的词**（「AI炼丹师」「药屋少女」）用的字在聊天记录里根本没见过。
    """

    haystack = _normalize(chat)
    charset = set(haystack)
    terms = _terms(content)
    suspicious: List[str] = []
    for term in terms:
        chars = [c for c in _normalize(term) if c.strip()]
        if not chars:
            continue
        covered = sum(1 for c in chars if c in charset)
        # 这个词的字有一半以上在聊天记录里没出现过 → 大概率是编的
        if covered / len(chars) < 0.5:
            suspicious.append(term)
    return (not suspicious), suspicious


def main() -> int:
    parser = argparse.ArgumentParser(description="拿记忆回聊天记录里核对")
    parser.add_argument("--db", required=True, help="memfade.db 路径")
    parser.add_argument("--messages", default="", help="MaiBot 的 MaiBot.db（消息库）")
    parser.add_argument("--chat-file", default="", help="纯文本聊天记录（与 --messages 二选一）")
    parser.add_argument("--scope", default="", help="只看某个作用域")
    parser.add_argument("--llm", action="store_true", help="用 LLM 逐条判断有没有依据")
    parser.add_argument(
        "--host-root", default="", help="MaiBot 根目录（--llm 时优先用它的 LLM）"
    )
    parser.add_argument(
        "--llm-base-url", default="", help="OpenAI 兼容接口地址（非 MaiBot 宿主时用）"
    )
    parser.add_argument("--llm-model", default="", help="模型名")
    parser.add_argument("--llm-api-key", default="", help="API Key")
    args = parser.parse_args()

    if args.messages:
        chat = load_chat(args.messages)
    elif args.chat_file:
        with open(args.chat_file, "r", encoding="utf-8", errors="replace") as handle:
            chat = handle.read()
    else:
        print("要给 --messages 或 --chat-file 其中一个")
        return 1

    conn = sqlite3.connect("file:%s?mode=ro" % args.db.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if args.scope:
            rows = conn.execute(
                "SELECT id, kind, content FROM memories WHERE scope_key = ? ORDER BY id",
                (args.scope,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, content FROM memories ORDER BY id"
            ).fetchall()
    finally:
        conn.close()

    print("聊天记录 %d 字，记忆 %d 条" % (len(chat), len(rows)))
    print("=" * 72)
    if args.llm:
        import asyncio

        if not args.host_root:
            print(
                "--llm 需要 --host-root（用 MaiBot 的 LLM）"
                " 或 --llm-base-url + --llm-model（用 OpenAI 兼容接口）"
            )
            return 1
        messages = load_messages(args.messages) if args.messages else chat.splitlines()
        verdicts = asyncio.run(
            _verify_with_llm(
                [dict(row) for row in rows],
                messages,
                args.host_root,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_api_key=args.llm_api_key,
            )
        )
        bad = 0
        for item in verdicts:
            mark = "OK " if "有依据" in item["verdict"] and "没依据" not in item["verdict"] else "[!]"
            if mark == "[!]":
                bad += 1
            print("  %s #%-4s %-8s %s" % (mark, item["id"], item["verdict"], item["note"][:40]))
            print("       %s" % (item["content"] or "")[:60])
        print("=" * 72)
        print("%d 条记忆在聊天里没有依据。" % bad)
        return 1 if bad else 0

    bad = 0
    for row in rows:
        ok, missing = verify(row["content"] or "", chat)
        if ok:
            continue
        bad += 1
        print(
            "  [!] #%-4s 这些词在聊天记录里找不到：%s"
            % (row["id"], "、".join(missing[:4]))
        )
        print("       %s" % (row["content"] or "")[:60])
    print("=" * 72)
    if bad:
        print("%d 条记忆在聊天记录里对不上——**可能是模型编的**。" % bad)
    else:
        print("每条记忆的话都能在聊天记录里找到。")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
