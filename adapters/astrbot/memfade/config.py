"""配置。

一份配置同时服务两种宿主：

* MaiBot 插件读 TOML（``config.toml``）；
* 其它框架的适配器读 Python dict 或 JSON。

所以配置对象只认 dict —— TOML/JSON 解析交给调用方，
核心库不绑定任何配置格式，也不绑定任何宿主。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional


@dataclass
class StorageConfig:
    db_path: str = "memfade.db"
    """SQLite 文件路径。相对路径按宿主提供的工作目录解析。"""

    enable_fts: bool = True
    """是否启用 FTS5 全文索引（中文二元组）。"""

    enable_vector: bool = True
    """是否启用向量检索。需要 embedding 配置有效。"""

    busy_timeout_ms: int = 5000


@dataclass
class ExtractionConfig:
    enabled: bool = True
    """是否从对话中抽取记忆。"""

    every_n_turns: int = 3
    """每隔多少轮触发一次抽取。"""

    min_messages: int = 2
    """一轮里至少有几条消息才值得抽取。"""

    min_importance: float = 0.35
    """低于这个显著性直接丢弃，不入库。"""

    min_importance_episodic: float = 0.20
    """一次性事件的单独门槛。

    事件类记忆天生重要性低（抽取器给 0.2~0.45），
    若和稳定事实用同一个门槛，它们会被全部滤掉——
    那淡化梯度就永远没机会上场。事件便宜、靠自然退场淘汰，
    所以门槛放低；稳定事实是常驻注入的，门槛保持高。
    """

    max_per_batch: int = 8
    """单次抽取最多写入几条，防止一次性灌爆。"""

    use_rules_fallback: bool = True
    """LLM 不可用时是否退化到规则抽取。"""

    promote_cross_scope: bool = True
    """是否允许抽取器把「用户主动要求记住」的内容提升为跨会话共享。"""


@dataclass
class RetrievalConfig:
    top_k: int = 8
    """默认召回条数。"""

    budget_tokens: int = 600
    """注入上下文的 token 预算。"""

    channel_candidates: int = 40
    """每条召回通道各取多少候选。"""

    min_score: float = 0.12
    """低于此分直接丢弃。"""

    echo_max: int = 3
    """回声层最多注入几条淡化记忆。"""

    max_item_chars: int = 120
    """单条记忆注入时的字数上限。超长截断——一段赛制说明塞进 prompt
    既费 token 又会淹掉其它记忆。"""

    max_profile_chars: int = 60
    """画像块里单条的字数上限。画像应该是短句（名字、喜好、忌讳）。"""

    echo_min_relevance: float = 0.35
    """回声层要求的最低文本相关度——只有被直接问起才带出来。"""

    w_relevance: float = 0.34
    w_importance: float = 0.20
    w_strength: float = 0.22
    w_recency: float = 0.14
    w_frequency: float = 0.10

    recency_tau_days: float = 7.0
    """新鲜度的衰减尺度（天）。"""

    strength_floor: float = 0.35
    """强度在总分里的下限系数：强度为 0 时仍保留 strength_floor 比例，
    避免一条刚被强化的重要事实因为强度稍低就被彻底压下去。"""

    association_boost: float = 0.05
    """与首条命中共享关键词的候选获得的加成。"""

    mmr_lambda: float = 0.25
    """MMR 多样性系数。

    一次注入里塞了好几条讲同一件事的，是「重复语段太多」的另一种来源
    （库里存了两份由去重解决）。这个系数决定「越像已选内容的候选，
    被压得越狠」。0 表示不启用。
    """

    dedup_threshold: float = 0.82
    """写入时的近似重复阈值（二元组包含度）。"""

    revive_min_common: int = 2
    """「被问起就想起」的门槛：查询与记忆的公共子串至少几个字。

    已归档的记忆平时不出现，但对方明确问到时应该能被捞回来。
    2 个字是一个有意义的中文词（「炸鸡」「上海」「老师」），
    再长会漏掉「还记得炸鸡吗」这种问法；再短会把「吃了」也算成问起。
    """

    merge_threshold: float = 0.92
    """整理阶段的合并阈值。"""

    reinject_recent: int = 4
    """无条件注入最近几条记忆条数（保持对话连续感）。"""


@dataclass
class FadeConfig:
    enabled: bool = True
    recall_alpha: float = 0.35
    reinforce_alpha: float = 0.8
    weaken_alpha: float = 0.5
    gist_enabled: bool = True
    """是否在淡出时把原文压缩成 gist。关掉则只降权，不做摘要。"""

    gist_max_chars: int = 40
    archive_after_days: int = 0
    """归档后多少天物理删除。0 表示永不删除（只归档）。"""

    recall_cooldown_seconds: int = 3600
    """强化冷却。同一条记忆在冷却期内被再次召回不重复强化，
    避免「一被检索就变强 → 更容易被检索」的正反馈把分数刷爆。"""


@dataclass
class MaintenanceConfig:
    enabled: bool = True
    interval_seconds: int = 3600
    """维护任务的最小间隔。"""

    max_rows_per_pass: int = 2000
    """单次维护最多处理多少条，避免长事务卡住。"""

    enable_merge: bool = True
    enable_gist: bool = True
    enable_supersede: bool = True

    max_memories_per_scope: int = 5000
    """单个会话的记忆条数上限。

    聊天机器人长年在线，记忆库会一直涨。超限后按
    「已归档 → 残影 → 淡出中 → 活跃」的顺序淘汰最弱的，
    置顶的永不淘汰。
    """


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 800
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    dim: int = 0
    """向量维度。0 表示首次调用时自动探测并记住。"""

    timeout: float = 30.0
    batch_size: int = 16
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MemFadeConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fade: FadeConfig = field(default_factory=FadeConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "MemFadeConfig":
        """从嵌套 dict 构造。未知键忽略，缺省值保留。"""

        config = cls()
        if not data:
            return config
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            current = getattr(config, f.name)
            if is_dataclass(current) and isinstance(value, dict):
                _apply_dict(current, value)
            else:
                setattr(config, f.name, value)
        return config

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value):
                out[f.name] = {g.name: getattr(value, g.name) for g in fields(value)}
            else:
                out[f.name] = value
        return out


def _apply_dict(target: Any, data: dict[str, Any]) -> None:
    """把 dict 里的已知键写进 dataclass 实例。"""

    known = {f.name for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(target, key, value)

