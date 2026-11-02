"""NoneBot2 插件配置。"""

from nonebot import get_plugin_config
from pydantic import BaseModel, Field


class Config(BaseModel):
    """memfade 配置。"""

    memfade_db_path: str = Field(default="memfade.db", description="SQLite 记忆库路径")
    memfade_top_k: int = Field(default=6, description="每轮最多召回多少条记忆")
    memfade_max_inject_tokens: int = Field(default=600, description="注入上下文的 token 预算")
    memfade_echo_max: int = Field(default=2, description="最多注入多少条已淡出的旧记忆")
    memfade_extract_enabled: bool = Field(default=True, description="是否从对话中抽取记忆")
    memfade_extract_every_n: int = Field(default=3, description="每隔多少条消息抽取一次")
    memfade_min_importance: float = Field(default=0.35, description="低于此显著性的内容不入库")
    memfade_llm_base_url: str = Field(default="", description="抽取用的 LLM 接口（留空则用规则抽取）")
    memfade_llm_api_key: str = Field(default="", description="抽取用的 LLM API Key")
    memfade_llm_model: str = Field(default="", description="抽取用的 LLM 模型名")
    memfade_self_names: list[str] = Field(
        default_factory=list,
        description="机器人自己的名字：主语是它时不入库",
    )
    memfade_focus_subjects: list[str] = Field(
        default_factory=list,
        description="只记这些主语的事（留空=不限制）",
    )
    memfade_subject_hint: str = Field(default="", description="抽取提示词里的人物提示，形如「（小许 / 季怡）」")


try:
    config = get_plugin_config(Config)
except Exception:
    # NoneBot 还没初始化时（插件被提前导入、或单测里）也要能用默认值，
    # 不能因为拿不到配置就整个插件加载失败。
    config = Config()

