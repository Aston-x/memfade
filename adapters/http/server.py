"""memfade · HTTP 侧车。

一个只用标准库写的本地 HTTP 服务，把 memfade 引擎暴露成 JSON 接口。

**它解决什么问题？**

memfade 的核心是 Python。但市面上的 QQ 机器人不止 Python：
Koishi 是 TypeScript，QQ 频道的 SDK 也有 Node 版本。
与其给每个语言重写一遍淡化算法（必然走样），不如让核心只跑一份，
其它语言通过 HTTP 调用它。

接口：

    GET  /health              健康检查
    POST /remember            写入记忆
    POST /recall              召回记忆
    POST /context             召回 + 分层组装（返回可直接注入的文本）
    POST /consolidate         跑一轮整理
    GET  /stats               统计

鉴权：请求头 ``X-MemFade-Token``，与服务端 ``--token`` 一致才放行。
默认只监听 127.0.0.1。

启动：

    python adapters/http/server.py --db memfade.db --port 8787 --token secret
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade import MemFadeConfig, MemoryDraft, MemoryKind, Scope  # noqa: E402
from memfade.engine import MemoryEngine  # noqa: E402
from memfade.types import Memory  # noqa: E402


def _scope_from(payload: Dict[str, Any]) -> Scope:
    """从请求体构造作用域。跨会话共享必须显式打开。"""

    raw = payload.get("scope") or {}
    if not isinstance(raw, dict):
        raw = {}
    return Scope(
        platform=str(raw.get("platform") or "qq"),
        chat_type=str(raw.get("chat_type") or "private"),
        chat_id=str(raw.get("chat_id") or "default"),
        cross_scope=bool(raw.get("cross_scope", False)),
    )


class MemFadeHTTPService:
    """业务逻辑：所有 HTTP handler 都调它。"""

    def __init__(self, engine: MemoryEngine) -> None:
        self.engine = engine
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def start_loop(self) -> None:
        """起一个常驻事件循环，供异步接口使用。"""

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="memfade-loop", daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread

    def stop_loop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3)
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def _run_async(self, coro: Any) -> Any:
        if self._loop is None:
            return asyncio.run(coro)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120)

    # -- 接口实现 ------------------------------------------------------ #

    def remember(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scope = _scope_from(payload)
        drafts_raw = payload.get("drafts")
        drafts: list[MemoryDraft] = []
        if isinstance(drafts_raw, list) and drafts_raw:
            for item in drafts_raw:
                if not isinstance(item, dict):
                    continue
                drafts.append(
                    MemoryDraft(
                        content=str(item.get("content") or "").strip(),
                        kind=_kind_of(item.get("kind")),
                        importance=float(item.get("importance") or 0.5),
                        subject=str(item.get("subject") or ""),
                        predicate=str(item.get("predicate") or ""),
                        object=str(item.get("object") or ""),
                        source=str(item.get("source") or "http"),
                        cross_scope=bool(item.get("cross_scope", False)),
                    )
                )
        else:
            content = str(payload.get("content") or "").strip()
            if content:
                drafts.append(
                    MemoryDraft(
                        content=content,
                        kind=_kind_of(payload.get("kind")),
                        importance=float(payload.get("importance") or 0.5),
                        subject=str(payload.get("subject") or ""),
                        predicate=str(payload.get("predicate") or ""),
                        object=str(payload.get("object") or ""),
                        source=str(payload.get("source") or "http"),
                        cross_scope=bool(payload.get("cross_scope", False)),
                    )
                )
        if not drafts:
            return {"ok": True, "written": 0, "ids": []}
        # 用 remember_atomic：太长的一条会按标点拆成「一件事一条」。
        # 侧车是给别的框架用的，写进来的东西多半没人再管——
        # 一段 200 字塞进去，注入时只能截断。
        written: List[Memory] = []
        for draft in drafts:
            written.extend(self.engine.remember_atomic(draft, scope))
        return {"ok": True, "written": len(written), "ids": [m.id for m in written]}

    def extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """从一段对话里抽取记忆（需要侧车配了 LLM）。

        为什么要有这个接口：``/remember`` 是「原样写进去」——
        别的框架（Koishi）拿它当抽取用，等于把整句聊天记录塞进记忆库，
        记忆库就成了聊天记录备份，而不是「关于这个人的事实」。
        配了 LLM 就走这里，让模型来提炼。
        """

        scope = _scope_from(payload)
        raw_messages = payload.get("messages")
        messages: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                entry = {"role": str(item.get("role") or "user"), "content": content}
                speaker = str(item.get("speaker") or "").strip()
                if speaker:
                    entry["speaker"] = speaker
                messages.append(entry)
        if not messages:
            return {"ok": True, "written": 0, "ids": [], "note": "没有可抽取的消息"}
        written = self._run_async(
            self.engine.remember_from_messages(
                messages,
                scope,
                source=str(payload.get("source") or "http-extract"),
            )
        )
        # mode 说明这批记忆是怎么来的：
        # - llm   模型抽的；
        # - rules 没配 LLM 时的规则兜底（**不是**把原话整句存进去）。
        # 调用方据此决定要不要退回到「原样写入」——
        # 而原样写入就是把聊天记录当记忆库，那是污染，不是记忆。
        return {
            "ok": True,
            "written": len(written),
            "ids": [m.id for m in written],
            "llm": self.engine.llm is not None,
            "mode": "llm" if self.engine.llm is not None else "rules",
        }

    def recall(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scope = _scope_from(payload)
        query = str(payload.get("query") or "")
        top_k = int(payload.get("top_k") or 6)
        # 这是显式检索：必须真的命中，不把「最近写入」的无关记忆混进来。
        # 查询为空时就是「随便看看」，这时不设门槛。
        hits = self.engine.recall(query, scope, top_k=top_k, require_match=bool(query))
        return {"ok": True, "hits": [_hit_to_dict(h) for h in hits]}

    def context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scope = _scope_from(payload)
        query = str(payload.get("query") or "")
        top_k = int(payload.get("top_k") or 6)
        ctx = self.engine.build_context(query, scope, top_k=top_k)
        return {
            "ok": True,
            "text": ctx.text,
            "token_estimate": ctx.token_estimate,
            "facts": [_hit_to_dict(h) for h in ctx.facts],
            "echoes": [_hit_to_dict(h) for h in ctx.echoes],
        }

    def consolidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scope = _scope_from(payload) if payload.get("scope") else None
        report = self._run_async(self.engine.consolidate(scope))
        return {"ok": True, "report": report.as_dict()}

    def forget(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """忘掉一条记忆（按 id 或按内容片段）。"""

        scope = _scope_from(payload)
        memory_id = payload.get("id")
        if memory_id:
            ok = self.engine.forget(int(memory_id))
            return {"ok": bool(ok), "forgotten": int(memory_id) if ok else None}
        content = str(payload.get("content") or "").strip()
        if not content:
            return {"ok": False, "error": "要给 id 或 content"}
        hits = self.engine.recall(content, scope, top_k=5, require_match=True)
        if not hits:
            return {"ok": False, "error": "没找到"}
        target = hits[0].memory
        self.engine.forget(target.id)
        return {"ok": True, "forgotten": target.id, "content": target.content}

    def pin(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """置顶/取消置顶一条记忆（置顶的不参与衰减与淘汰）。"""

        scope = _scope_from(payload)
        memory_id = payload.get("id")
        pinned = bool(payload.get("pinned", True))
        if memory_id:
            ok = self.engine.pin(int(memory_id), pinned=pinned)
            return {"ok": bool(ok), "id": int(memory_id), "pinned": pinned}
        content = str(payload.get("content") or "").strip()
        if not content:
            return {"ok": False, "error": "要给 id 或 content"}
        hits = self.engine.recall(content, scope, top_k=5, require_match=True)
        if not hits:
            return {"ok": False, "error": "没找到"}
        target = hits[0].memory
        ok = self.engine.pin(target.id, pinned=pinned)
        return {"ok": bool(ok), "id": target.id, "pinned": pinned, "content": target.content}

    def repair(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """改掉记错的那条记忆。

        抽取难免出错（把「今天」写成「每天」）。出错之后得有个补救口——
        自己改一条，比整库清空的代价小得多。
        """

        scope = _scope_from(payload)
        old = str(payload.get("old") or payload.get("content") or "").strip()
        new = str(payload.get("new") or "").strip()
        if not old or not new:
            return {"ok": False, "error": "要给 old 与 new"}
        updated = self._run_async(self.engine.repair(scope, old, new))
        if updated is None:
            return {"ok": False, "error": "没找到"}
        return {
            "ok": True,
            "id": updated.id,
            "content": updated.content,
            "previous": str((updated.meta or {}).get("previous_content") or ""),
        }

    def stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scope = _scope_from(payload) if payload.get("scope") else None
        return {"ok": True, "stats": self.engine.stats(scope)}

    def list_memories(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """列出某个作用域里最近记下的内容。

        为什么单独开一个接口：只给数字（「记住了 12 条」）没人能判断记忆库干不干净，
        得**看得见具体内容**才知道有没有记错东西。
        """

        scope = _scope_from(payload)
        limit = max(1, min(50, int(payload.get("limit") or 10)))
        try:
            ids = [mid for mid, _ in self.engine.store.recent(scope.key(), limit=limit)]
            memories = sorted(
                self.engine.store.get_many(ids), key=lambda m: m.created_at, reverse=True
            )
        except Exception:
            memories = []
        return {
            "ok": True,
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "kind": m.kind.value,
                    "status": m.status.value,
                    "importance": round(m.importance, 3),
                    "created_at": m.created_at,
                }
                for m in memories
            ],
        }


def _kind_of(value: Any) -> MemoryKind:
    try:
        return MemoryKind(str(value or "").lower())
    except ValueError:
        return MemoryKind.EPISODIC


def _hit_to_dict(hit: Any) -> Dict[str, Any]:
    mem = hit.memory
    return {
        "id": mem.id,
        "content": mem.content,
        "summary": mem.summary,
        "kind": mem.kind.value,
        "status": mem.status.value,
        "tier": hit.tier,
        "score": hit.score,
        "relevance": hit.relevance,
        "strength": hit.strength_eff,
        "importance": mem.importance,
        "recall_count": mem.recall_count,
        "created_at": mem.created_at,
        "scope_key": mem.scope_key,
    }


class MemFadeHandler(BaseHTTPRequestHandler):
    """HTTP 入口。"""

    service: MemFadeHTTPService
    token: str = ""
    server_version = "memfade/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[memfade-http] " + (fmt % args) + "\n")

    def _authorized(self) -> bool:
        if not self.token:
            return True
        return self.headers.get("X-MemFade-Token", "") == self.token

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        if not self._authorized():
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._send(200, {"ok": True, "service": "memfade", "version": "0.1.0"})
            return
        if path == "/stats":
            payload = self._read_query_payload()
            self._send(200, self.service.stats(payload))
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        payload = self._read_json()
        try:
            if path == "/remember":
                self._send(200, self.service.remember(payload))
            elif path == "/extract":
                self._send(200, self.service.extract(payload))
            elif path == "/recall":
                self._send(200, self.service.recall(payload))
            elif path == "/context":
                self._send(200, self.service.context(payload))
            elif path == "/consolidate":
                self._send(200, self.service.consolidate(payload))
            elif path == "/forget":
                self._send(200, self.service.forget(payload))
            elif path == "/pin":
                self._send(200, self.service.pin(payload))
            elif path == "/repair":
                self._send(200, self.service.repair(payload))
            elif path == "/list":
                self._send(200, self.service.list_memories(payload))
            elif path == "/stats":
                # GET 也能取（浏览器直接打开就能看），POST 同样受理——
                # 前端框架（Koishi 等）默认用 POST。
                self._send(200, self.service.stats(payload))
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._send(500, {"ok": False, "error": str(exc)})

    def _read_query_payload(self) -> Dict[str, Any]:
        """把 URL 查询参数转成 payload。"""

        if "?" not in self.path:
            return {}
        from urllib.parse import parse_qs

        query = parse_qs(self.path.split("?", 1)[1])
        out: Dict[str, Any] = {}
        for key, values in query.items():
            if not values:
                continue
            out[key] = values[0]
        if "scope" in out and isinstance(out["scope"], str):
            try:
                out["scope"] = json.loads(out["scope"])
            except json.JSONDecodeError:
                out["scope"] = {"chat_id": out["scope"]}
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="memfade HTTP 侧车")
    parser.add_argument("--db", default="memfade.db", help="SQLite 记忆库路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8787, help="监听端口")
    parser.add_argument("--token", default="", help="鉴权令牌（X-MemFade-Token）")
    parser.add_argument(
        "--embedding-url",
        default="",
        help="OpenAI 兼容的 embeddings 接口地址（留空则不用向量检索）",
    )
    parser.add_argument("--embedding-model", default="", help="嵌入模型名")
    parser.add_argument("--embedding-key", default="", help="嵌入接口的 API key")
    parser.add_argument(
        "--llm-url",
        default="",
        help="OpenAI 兼容的 chat 接口地址（留空则只能原样写入，不能抽取）",
    )
    parser.add_argument("--llm-model", default="", help="抽取用的模型名")
    parser.add_argument("--llm-key", default="", help="LLM 接口的 API key")
    args = parser.parse_args()

    config = MemFadeConfig.from_dict({"storage": {"db_path": args.db}})
    embedder = None
    if args.embedding_url and args.embedding_model:
        from memfade.embedding import HttpEmbeddingClient

        embedder = HttpEmbeddingClient(
            base_url=args.embedding_url,
            api_key=args.embedding_key,
            model=args.embedding_model,
        )
    engine = MemoryEngine(config, embedder=embedder)
    if args.llm_url and args.llm_model:
        from memfade.llm import HttpLLMClient

        engine.llm = HttpLLMClient(
            base_url=args.llm_url,
            api_key=args.llm_key,
            model=args.llm_model,
        )
        engine.extractor.llm = engine.llm
        sys.stderr.write("[memfade-http] 已接上 LLM：%s\n" % args.llm_model)
    service = MemFadeHTTPService(engine)
    service.start_loop()

    handler = type("BoundHandler", (MemFadeHandler,), {"service": service, "token": args.token})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    sys.stderr.write(
        f"[memfade-http] 监听 http://{args.host}:{args.port}  库={os.path.abspath(args.db)}\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        service.stop_loop()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


