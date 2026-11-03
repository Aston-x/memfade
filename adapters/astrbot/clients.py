"""把 AstrBot 的 Provider 能力包装成 memfade 的协议。

memfade 核心不绑定任何厂商 SDK，只定义两个极小的协议（``memfade/llm.py``、
``memfade/embedding.py``）：

* ``LLMClient.complete(prompt, *, system, max_tokens, temperature) -> str``
* ``EmbeddingClient.embed(texts) -> list[list[float]]``

这里各写一个薄适配类，复用宿主已经配好的 provider——客户不需要再填一份
API key，也不会出现「插件里的模型和面板里不是一个」的割裂。

版本兼容
--------
宿主有两代 API，这里都认：

* 4.29.4：``Context.get_using_provider_async(umo)``（异步），``get_using_provider``
  已标 deprecated
* 4.14.6：只有同步的 ``Context.get_using_provider(umo)``

所以一律用 ``getattr`` 探测，先异步后同步，都没有就退回已配置的第一个 provider。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


class AstrBotLLMClient:
    """memfade 的 ``LLMClient`` 协议实现。

    注意：``Provider.text_chat`` 不接受 ``max_tokens`` / ``temperature``
    （4.14.6 的 openai_source 连 ``**kwargs`` 都不透传），所以这两个参数
    只在签名上对齐协议、实际忽略——采样参数由宿主 provider 自己决定。
    """

    def __init__(
        self,
        plugin: Any,
        provider: Any = None,
        provider_id: str = "",
        fallback: Any = None,
    ) -> None:
        self._plugin = plugin
        self._fixed = provider
        self._provider_id = provider_id
        #: 宿主 provider 不可用时的兜底（OpenAI 兼容接口）
        self._fallback = fallback

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        provider = self._fixed
        if provider is None:
            provider = await self._plugin.resolve_provider(self._provider_id)
        if provider is None:
            if self._fallback is not None:
                return await self._fallback.complete(
                    prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            raise RuntimeError("AstrBot 没有可用的 LLM provider")

        kwargs = {}
        if system:
            kwargs["system_prompt"] = system
        resp = await provider.text_chat(prompt=prompt, **kwargs)

        # LLMResponse.completion_text 是 property（4.14.6 / 4.29.4 都是）
        text = getattr(resp, "completion_text", "")
        if not str(text or "").strip():
            raise RuntimeError("provider 返回空结果")
        return str(text)


class AstrBotEmbeddingClient:
    """memfade 的 ``EmbeddingClient`` 协议实现。

    ``EmbeddingProvider`` 没有从 ``astrabot.api.provider`` 的公开导出里暴露
    （4.14.6 的 ``__all__`` 里没有它），所以这里不 import 那个类，直接鸭子类型
    调用 ``get_embeddings()``——跨版本都不会 ImportError。
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        items = [str(t) for t in (texts or [])]
        if not items:
            return []

        vectors = await self._provider.get_embeddings(items)
        out: List[List[float]] = []
        for vec in vectors or []:
            out.append([float(v) for v in (vec or [])])
        # provider 少返回时补空，避免与输入错位
        while len(out) < len(items):
            out.append([])
        return out[: len(items)]


async def resolve_chat_provider(context: Any, umo: str = "") -> Optional[Any]:
    """拿当前会话用的 chat provider。

    先异步版（4.29.4），再同步版（4.14.6），都没有就退回已配置的第一个。
    """
    getter = getattr(context, "get_using_provider_async", None)
    if getter is not None:
        try:
            provider = await getter(umo)
            if provider is not None:
                return provider
        except Exception:
            pass

    getter = getattr(context, "get_using_provider", None)
    if getter is not None:
        try:
            provider = getter(umo)
            if provider is not None:
                return provider
        except Exception:
            pass

    lister = getattr(context, "get_all_providers", None)
    if lister is not None:
        try:
            providers = lister() or []
            if providers:
                return providers[0]
        except Exception:
            pass
    return None


def resolve_embedding_provider(context: Any, provider_id: str = "") -> Optional[Any]:
    """挑一个 embedding provider。

    ``provider_id`` 非空时按 id 精确匹配；否则取第一个可用的。
    """
    lister = getattr(context, "get_all_embedding_providers", None)
    if lister is None:
        return None
    try:
        providers = list(lister() or [])
    except Exception:
        return None
    if not providers:
        return None

    if provider_id:
        for provider in providers:
            cfg = getattr(provider, "provider_config", None) or {}
            try:
                if str(cfg.get("id") or "") == str(provider_id):
                    return provider
            except Exception:
                continue
        return None
    return providers[0]


def provider_label(provider: Any) -> str:
    """provider 的可读名字，用在「记忆 状态」的诊断输出里。"""
    if provider is None:
        return ""
    cfg = getattr(provider, "provider_config", None) or {}
    for key in ("id", "model", "type"):
        value = ""
        try:
            value = str(cfg.get(key) or "")
        except Exception:
            value = ""
        if value:
            return value
    meta = getattr(provider, "meta", None)
    if meta is not None:
        try:
            return str(getattr(meta, "id", "") or "")
        except Exception:
            return ""
    return type(provider).__name__
