"""向量嵌入层。

和 LLM 层一样，核心只定义协议。默认实现走 OpenAI 兼容的
``/embeddings`` 接口（百炼 text-embedding-v3、DeepSeek、bge-m3 服务、
本地 Ollama / vLLM 都能用）。

**没有 embedding 也能跑**：检索会退化成纯 BM25 + 近期性，
只是语义召回变弱，不影响淡化、隔离、去重这些核心能力。
这很重要——很多聊天机器人跑在内存紧张的机器上，
塞一个本地 embedding 模型进去往往先把它压垮。
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """最小嵌入协议。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把若干段文本变成向量。返回长度必须与入参一致。"""


@dataclass
class HttpEmbeddingClient:
    """OpenAI 兼容的 embeddings 客户端。"""

    base_url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 30.0
    batch_size: int = 16
    extra_headers: dict[str, str] = field(default_factory=dict)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.base_url or not self.model:
            raise RuntimeError("Embedding 未配置（需要 base_url 与 model）")
        items = list(texts)
        if not items:
            return []
        out: list[list[float]] = []
        for start in range(0, len(items), max(1, self.batch_size)):
            chunk = items[start : start + max(1, self.batch_size)]
            payload = {"model": self.model, "input": chunk}
            url = self.base_url.rstrip("/") + "/embeddings"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers.update(self.extra_headers or {})
            def _post() -> dict:
                request = urllib.request.Request(
                    url, data=json.dumps(payload).encode("utf-8"), headers=headers
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))

            body = await asyncio.to_thread(_post)
            data = body.get("data") or []
            # 某些实现不保证顺序，按 index 归位
            ordered: list[Optional[list[float]]] = [None] * len(chunk)
            for item in data:
                idx = int(item.get("index", 0))
                if 0 <= idx < len(ordered):
                    ordered[idx] = [float(v) for v in (item.get("embedding") or [])]
            for vec in ordered:
                out.append(vec or [])
        return out


class CallableEmbeddingClient:
    """把宿主提供的函数包装成 EmbeddingClient。"""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._fn(list(texts))
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        return [[float(v) for v in vec] for vec in result]


class NullEmbeddingClient:
    """什么都不做的实现——用于明确关闭向量的场合。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[] for _ in texts]


def make_embedder(
    client: Optional[EmbeddingClient],
) -> Optional[Callable[[str], Any]]:
    """把 EmbeddingClient 包成「查一条」的便捷函数；没有客户端则返回 None。

    返回的是**异步**函数——宿主多半是 async 的，强行同步只会更难接。
    """

    if client is None:
        return None

    async def _embed_one(text: str) -> Optional[list[float]]:
        try:
            vectors = await client.embed([text])
        except Exception:
            return None
        if not vectors or not vectors[0]:
            return None
        return list(vectors[0])

    return _embed_one

