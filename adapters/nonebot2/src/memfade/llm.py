"""LLM 调用层。

核心库不绑定任何厂商 SDK，只定义一个极小的协议；
默认实现走标准库的 ``urllib`` 调 OpenAI 兼容接口（DashScope /
DeepSeek / Kimi / OpenAI / 本地 vLLM 都能用）。

宿主也可以直接塞一个 callable 进来（见 ``CallableLLMClient``），
这样 MaiBot 这类自带 LLM 调度的宿主可以直接复用自己的通道，
不必让插件再单独配一份 API key。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """最小 LLM 协议。"""

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        """返回模型的纯文本输出。失败时抛异常，由调用方兜底。"""


@dataclass
class HttpLLMClient:
    """OpenAI 兼容的 chat completions 客户端。"""

    base_url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 30.0
    temperature: float = 0.2
    extra_headers: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra_headers is None:
            self.extra_headers = {}

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        if not self.base_url or not self.model:
            raise RuntimeError("LLM 未配置（需要 base_url 与 model）")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature if temperature is not None else self.temperature),
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers or {})

        def _post() -> dict:
            request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        body = await asyncio.to_thread(_post)
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 返回空结果")
        return str(choices[0].get("message", {}).get("content") or "")


class CallableLLMClient:
    """把宿主提供的同步/异步函数包装成 LLMClient。"""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        result = self._fn(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        return str(result or "")


def parse_json_block(text: str) -> Optional[Any]:
    """从模型输出里抠出 JSON。

    模型经常在 JSON 外面裹一段说明或 ```json 围栏，这里逐层剥离。
    """

    if not text:
        return None
    raw = str(text).strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 退一步：找第一个 { 或 [ 到最后一个 } 或 ]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None

