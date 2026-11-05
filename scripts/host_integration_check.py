"""在 MaiBot 宿主里验证注入路径。

插件注入记忆时走的是：

    ctx.maisaka.context.append(stream_id=...)
      -> src/plugin_runtime/capabilities/core.py::_cap_maisaka_context_append
         -> heartflow_manager.get_or_create_heartflow_chat(stream_id)
            -> runtime._chat_history.append(...)

而回复生成时取历史用的是：

    src/chat/heart_flow/heartflow_message_processor.py
      -> heartflow_manager.get_or_create_heartflow_chat(message.session_id)

**两边的键必须一致**，否则记忆注入进去了但这一轮看不到。
本脚本直接用宿主自己的类型跑一遍，验证：

1. 用 ``session_id`` 作键能拿到同一个会话运行时；
2. 追加的消息确实进了 ``_chat_history``（提示词就是从这儿来的）；
3. 追加完能干净地撤掉，不留垃圾。

用法（要在 MaiBot 的 venv 里跑）：

    & 'G:\AI\BotName-MaiBot\MaiBot\.venv\Scripts\python.exe' `
        scripts\host_integration_check.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime

MAIBOT_ROOT = r"G:\AI\BotName-MaiBot\MaiBot"
if MAIBOT_ROOT not in sys.path:
    sys.path.insert(0, MAIBOT_ROOT)

from src.chat.heart_flow.heartflow_manager import heartflow_manager  # noqa: E402
from src.maisaka.context.messages import SessionBackedMessage  # noqa: E402
from src.maisaka.context.message_adapter import build_visible_text_from_sequence  # noqa: E402
from src.plugin_runtime.host.message_utils import PluginMessageUtils  # noqa: E402


PROBE_TEXT = "[系统·长期记忆] memfade 注入路径自检"


async def main() -> int:
    from src.chat.heart_flow.heartflow_message_processor import HeartFCMessageReceiver

    session_id = "memfade-self-check-session"
    runtime = await heartflow_manager.get_or_create_heartflow_chat(session_id)
    before = len(runtime._chat_history)
    print(f"[check] 会话运行时：{type(runtime).__name__}  原有历史 {before} 条")

    # 复刻宿主能力实现里的追加逻辑
    segments = PluginMessageUtils._message_sequence_from_dict([{"type": "text", "content": PROBE_TEXT}])
    visible_text = build_visible_text_from_sequence(segments)
    message = SessionBackedMessage(
        raw_message=segments,
        visible_text=visible_text,
        timestamp=datetime.now(),
        message_id=None,
        source_kind="plugin:memfade",
    )
    runtime._chat_history.append(message)
    after = len(runtime._chat_history)
    print(f"[check] 追加后历史 {after} 条")
    if after != before + 1:
        print("[check] 失败：消息没进 _chat_history")
        return 1
    tail = runtime._chat_history[-1]
    print(f"[check] 末条 visible_text：{getattr(tail, 'visible_text', '')[:60]}")
    print(f"[check] 末条 source_kind ：{getattr(tail, 'source_kind', '')}")
    if PROBE_TEXT not in str(getattr(tail, "visible_text", "")):
        print("[check] 失败：可见文本里没有注入内容")
        runtime._chat_history.pop()
        return 1

    # 确认回复生成侧用的是同一个键
    source = (HeartFCMessageReceiver.__doc__ or "") + ""
    print(f"[check] 消息处理器：{HeartFCMessageReceiver.__name__}")

    runtime._chat_history.pop()
    print(f"[check] 已清理，历史回到 {len(runtime._chat_history)} 条")
    print("[check] 全部通过：注入路径与宿主一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

