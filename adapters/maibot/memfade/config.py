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

    min_importance: float = 0.45
    """低于这个显著性直接丢弃，不入库。

    为什么从 0.35 抬到 0.45：稳定事实是**常驻注入**的，
    门槛低了，模型随口一句「今天只有一节课」就能变成
    「每天只有一节课」，陪着机器人说一年。

    ⚠ 门槛卡的是**模型自己给的分数**，不是夹紧后的分数。
    种类区间（semantic 下限 0.40）只在写入时夹一次，
    提前夹会把模型给的 0.1 抬到 0.40，等于给垃圾开后门。
    """

    min_importance_episodic: float = 0.20
    """一次性事件的单独门槛。

    事件类记忆天生重要性低（抽取器给 0.2~0.45），
    若和稳定事实用同一个门槛，它们会被全部滤掉——
    那淡化梯度就永远没机会上场。事件便宜、靠自然退场淘汰，
    所以门槛放低；稳定事实是常驻注入的，门槛保持高。
    """

    store_evidence: bool = True
    """是否给每条记忆存「依据原话」。

    依据存在 ``meta["quote"]`` 里：面板上能看见这条记忆是从哪句话来的，
    事后核对、追责、修正都有据可依。
    写入前的闸门（``memfade.guards``）也要用它——
    没有依据就查不了「记忆里的断言是不是模型自己加的」。
    """

    evidence_max_chars: int = 120
    """依据原话最多存多少字。长了就没法一眼看完。"""

    max_per_batch: int = 8
    """单次抽取最多写入几条，防止一次性灌爆。"""

    max_chars: int = 40
    """一条记忆的最长字数。超了就按标点拆成「一件事一条」。

    提示词里已经要求「30 字以内」，但**提示词不是保证**——
    模型偶尔会甩回来一整段。超长记忆塞进库里，注入时只能截断
    （断在半句里），矛盾检测也抓不住，体检还会一直报警。
    """

    use_rules_fallback: bool = True
    """LLM 不可用时是否退化到规则抽取。"""

    promote_cross_scope: bool = True
    """是否允许抽取器把「用户主动要求记住」的内容提升为跨会话共享。"""

    subject_hint: str = ""
    """抽取提示词里的「这次对话里有谁」提示。

    核心库不绑定任何人名——部署时把名字填进来（形如「（许杏玖 / 季怡）」），
    抽取器才知道该用谁当主语。留空则用中性示例。
    """

    self_names: tuple[str, ...] = ()
    """机器人自己的名字（可以有多个别名）。

    抽取结果的主语是它时一律丢弃。理由很实在：**记忆库是「关于对方」的**，
    机器人自己的事（自己看过什么番、自己收藏了什么）记进去没有意义，
    还会让机器人对着主人认领别人的记忆。
    """

    focus_subjects: tuple[str, ...] = ()
    """只记这些主语的事。留空 = 不限制。

    群聊里认识很多人时用它收窄：主人叫「季怡」，那就只记季怡的事。
    """

    require_person_subject: bool = True
    """主语必须是「人」（对话里出现过的说话人，或常见的亲属/身份称呼）。

    非人的主语——《牧神记5》、第18届蓝桥杯、四个 t.wiki——
    都是知识条目或资料碎片，不是「关于某个人」的记忆，一律不入库。
    """

    strip_quotes: bool = True
    """抽取前是否剥掉引用与转发内容（由适配器实现）。

    这是**污染的最大来源**：宿主把「被引用的原文」拼进用户的消息正文，
    抽取器于是把机器人自己说过的话、群公告的原文，
    当成用户的话记了下来。
    """


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

    min_relevance: float = 0.05
    """注入时的相关性底线。

    检索有三条通道：全文、向量、近期写入。前两条命中才算「相关」，
    纯靠「最近写入」混进来的候选相关性为 0——把它们注入 prompt，
    就是「总提旧事」。默认要求至少有检索命中。
    """

    echo_max: int = 3
    """回声层最多注入几条淡化记忆。"""

    max_item_chars: int = 120
    """单条记忆注入时的字数上限。超长截断——一段赛制说明塞进 prompt
    既费 token 又会淹掉其它记忆。"""

    max_profile_chars: int = 60
    """画像块里单条的字数上限。画像应该是短句（名字、喜好、忌讳）。"""

    echo_min_relevance: float = 0.35
    """回声层要求的最低文本相关度——只有被直接问起才带出来。"""

    uni_discount: float = 0.45
    """单字兜底通道的相关度折价。

    单字命中（「辣」对上「喜欢辣的食物」）比多字词命中弱得多，
    折价之后照样能召回，但绝不会排在词面精确命中前面。
    """

    uni_max_query_chars: int = 4
    """只有这么短的查询才允许走单字兜底（按**实义字**个数算，虚词不算）。

    「辣」「猫猫」是关键词，值得兜底；「我记得你之前说过喜欢吃辣」
    是整句，兜底只会把无关记忆拖进来。

    ⚠ 改成 4 是因为「吃什么」这种三字问句要能命中「喜欢辣的食物」——
    词面只搭上一个「吃」字，没有向量通道时全靠这一条通道。
    「她喜欢吃什么」（五个实义字）仍然不放行：那种问句里的字太常见，
    放进来会拖出一堆无关记忆。
    """

    uni_max_hits: int = 20
    """单字命中这么多条，说明这个字太常见（「我」「的」），没有区分度。

    这种情况下宁可不用单字通道——让召回交给向量通道或干脆空着。
    """

    recent_rescue: bool = True
    """「刚聊过的那件事」的兜底要不要开。

    「最近写入」通道是**无条件**参与候选的：没有这道闸门，
    每一轮都会把最近几条塞进上下文——那就是「总提旧事」。
    但中文里「那部番叫什么」这种指代满地都是，
    词面检索一个都命中不了，答案往往就是刚刚聊过的那条。

    所以开着，但**限死**：只在查询明显在指代时，放一条进来。
    """

    recent_rescue_window_hours: float = 4.0
    """「刚聊过」的时限（小时）。超过这个时间的记忆不算「刚才那件事」。"""

    vector_min_gap: float = 0.15
    """向量通道的命中门槛：要比「这批记忆之间的典型相似度」高出多少。

    嵌入模型对**任何**文本都有 0.4~0.55 的基础相似度
    （本机 text-embedding-v3 实测）。不设门槛的话，
    哪怕用户说「今天天气真好啊」也会捞回一批记忆塞进上下文——
    那正是「总提旧事」的老毛病。

    取「中位数 + 门槛」而不是固定的余弦值：不同模型的基准相似度差很远
    （有的模型什么文本都是 0.8），**相对于本批记忆的中位数**才跨模型可用。

    实测（40 条真实记忆）：
      相关查询的「最佳 − 中位」= 0.13~0.41
      无关查询的「最佳 − 中位」= 0.08~0.14
    """

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
        elif isinstance(current, tuple) and isinstance(value, list):
            # TOML/JSON 读出来的是 list，配置里声明的是 tuple——统一一下，
            # 不然 `name in self_names` 这种按元素比较的地方会静默失配。
            setattr(target, key, tuple(value))
        else:
            setattr(target, key, value)

