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


config = get_plugin_config(Config)

