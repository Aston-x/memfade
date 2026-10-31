"""memfade · NoneBot2 适配器。

NoneBot2 没有统一的「往 LLM prompt 里插东西」的钩子——
多数 LLM 插件是自己组装 prompt 再调接口。所以本插件的定位是
**记忆服务 + 助手 API**：

* 自动旁听消息并存进记忆库（不阻塞任何其它插件）；
* 提供 ``/记忆``、``/记住``、``/忘掉`` 命令；
* 提供给其它插件直接调用的公开函数：

      from nonebot_plugin_memfade import build_context, remember, engine

      text = await build_context(event)      # 拿到可直接注入 prompt 的记忆文本
      await remember(event, "他叫许杏玖")      # 显式写入一条记忆

另外也会把记忆文本挂到 ``event.set_extra("memfade_context", ...)``，
想自动接管的插件直接读这个 extra 即可。
"""

from __future__ import annotations

from nonebot import require
from nonebot.plugin import PluginMetadata

require("nonebot_plugin_apscheduler")

from .config import Config  # noqa: E402

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-memfade",
    description="记忆淡化引擎：会记得，也会忘得恰到好处。艾宾浩斯衰减 + 会话严格隔离。",
    usage="https://github.com/Aston-x/memfade",
    type="application",
    homepage="https://github.com/Aston-x/memfade",
    config=Config,
    supported_adapters={"~onebot.v11", "~onebot.v12"},
)

from .core import build_context, engine, remember, recall  # noqa: E402,F401
from . import handlers  # noqa: E402,F401

