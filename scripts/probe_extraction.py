"""探针：用**宿主真实的 LLM** 跑一遍抽取，看看这段对话会被记成什么。

为什么不直接看线上库：抽取的提示词、纪律、配置都可能在改，
光看库里已有的条目看不出「下一次会记什么」。

这个脚本按部署时的那份 config.toml 构造抽取器，
所以它回答的是「线上真会怎么记」，不是「理想情况下应该怎么记」。

用法：

    python scripts/probe_extraction.py --host-root "G:/AI/许杏玖-MaiBot/MaiBot" \
        --plugin-config "G:/AI/许杏玖-MaiBot/MaiBot/plugins/memfade-plugin/config.toml" \
        --conversation "季怡：我在想怎么在小红书上卖AI头像赚钱养你呢"

对话可以写成多行，每行「说话人：内容」；
说话人留空就用「用户」。**引号与转发要先剥掉**——
适配器在实机上是这么做的，探针也一样（不然测的是没剥干净的文本）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memfade.config import ExtractionConfig  # noqa: E402
from memfade.extract import MemoryExtractor  # noqa: E402


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 老 Python
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _extraction_config(plugin_config: str) -> ExtractionConfig:
    data: Dict[str, Any] = {}
    if plugin_config and os.path.exists(plugin_config):
        raw = _load_toml(plugin_config)
        memory = raw.get("memory") or {}
        data = {
            "subject_hint": str(memory.get("subject_hint") or ""),
            "self_names": tuple(memory.get("self_names") or ()),
            "focus_subjects": tuple(memory.get("focus_subjects") or ()),
            "require_person_subject": bool(memory.get("require_person_subject", True)),
        }
    return ExtractionConfig(
        subject_hint=str(data.get("subject_hint") or ""),
        self_names=tuple(data.get("self_names") or ()),
        focus_subjects=tuple(data.get("focus_subjects") or ()),
        require_person_subject=bool(data.get("require_person_subject", True)),
    )


class _HostLLM:
    """借宿主的 LLM 服务用一下（和插件走同一条路）。"""

    def __init__(self, host_root: str, task_name: str = "memory") -> None:
        if host_root and host_root not in sys.path:
            sys.path.insert(0, host_root)
        from src.services import llm_service  # type: ignore

        self._service = llm_service
        self._task_name = task_name

    async def complete(self, prompt: str, *, system: str = "", max_tokens: int = 4000, temperature: float = 0.1) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request = self._service.LLMServiceRequest(
            task_name=self._task_name,
            request_type="memfade.probe",
            prompt=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = await self._service.generate(request)
        payload = result.to_capability_payload()
        return str(payload.get("response") or "")


def _messages(conversation: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in conversation.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, sep, content = line.partition("：")
        if not sep:
            speaker, content = "用户", line
        out.append(
            {
                "role": "user",
                "content": content.strip(),
                "speaker": speaker.strip() or "用户",
            }
        )
    return out


async def _run(args: argparse.Namespace) -> int:
    import asyncio

    config = _extraction_config(args.plugin_config)
    extractor = MemoryExtractor(config, _HostLLM(args.host_root, args.task_name))
    messages = _messages(args.conversation)
    result = await extractor.extract(messages, source="probe")

    print("=" * 66)
    print("对话（%d 条）" % len(messages))
    print("=" * 66)
    for msg in messages:
        print("  %s：%s" % (msg["speaker"], msg["content"][:60]))
    print("\n抽取结果：%d 条留下，%d 条丢弃" % (len(result.drafts), len(result.dropped)))
    print("-" * 66)
    for draft in result.drafts:
        print(
            "  留  [%s] %s  （%s / %s / %s）"
            % (
                draft.kind.value,
                draft.content,
                draft.subject or "-",
                draft.predicate or "-",
                draft.object or "-",
            )
        )
    for content, reason in result.dropped:
        print("  丢  [%s] %s" % (reason, content))
    if result.error:
        print("\n（抽取走了兜底：%s）" % result.error)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="用宿主 LLM 探一遍抽取")
    parser.add_argument("--conversation", required=True, help="多行「说话人：内容」")
    parser.add_argument("--host-root", required=True, help="MaiBot 根目录")
    parser.add_argument("--plugin-config", default="", help="部署时那份 config.toml")
    parser.add_argument("--task-name", default="memory", help="宿主模型任务名")
    args = parser.parse_args()
    import asyncio

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
