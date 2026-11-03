"""写入前的闸门：给候选记忆找依据，再决定它该不该入库。

为什么要有这一层
================

「记忆污染比记太多更严重」——一条编出来的事实会按稳定事实活很久，
甚至因为被反复注入而活得更久。实测里最常见的两种错法是：

1. **把这一次说成常态**：「今天只有一节课」→「每天只有一节课」；
2. **把当次的请求说成长期偏好**：「这次你用平时跟别人说话的语气」
   →「希望小玖用平时与别人说话的语气」。

提示词能压住大部分，但压不住全部——模型给错的东西往往还很有自信
（「每天只有一节课」它给 0.6 分，「希望小玖用…语气」它给 0.75 分），
而「阈值」这种纯数值门槛对它们一点用都没有。

所以闸门写在代码里，两道机制：

* **依据原话**：每条候选都回对话里找它的出处（用户自己说的话），
  存进 ``meta["quote"]``。面板上能看见「这条是从哪句话来的」，
  事后核对、追责、修正都有据可依。
* **准入判断**：拿依据跟内容对账。内容里冒出依据里没有的断言
  （「每天」「永远」「希望」），说明模型自己加戏了。

降级而不是直接丢
================

多数情况是把「稳定事实」降成「事件」：信息没丢，但它不再常驻注入，
过一阵子自己淡掉——这正是淡化引擎该干的活。
只有**临时状态**与**临时情绪**才直接丢：它们既不该常驻，
也不值得等它慢慢淡掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from .textutil import similarity, token_set
from .types import MemoryDraft, MemoryKind


# ---------------------------------------------------------------------- #
# 词表
# ---------------------------------------------------------------------- #

# 一次性时间词：说的是「这一次」，不能当成稳定事实。
TIME_MARKERS = (
    "今天", "今日", "今早", "今晚", "今儿", "现在", "目前", "眼下",
    "刚才", "刚刚", "方才", "当时", "那时", "这次", "这一次",
    "那次", "那天", "当天", "整天", "前天", "昨天", "昨日", "前几天",
    "前两天", "这几天", "最近", "近期", "前阵子", "上回", "这个月",
    "本月", "上个月", "今年", "去年", "明年", "这会儿",
)

# 常态词：明说了这是长期状态，带这类词的内容不受「时效」降级。
PERMANENCE_MARKERS = (
    "每天", "天天", "总是", "一直", "从来", "从不", "以后", "每次", "每回",
    "通常", "一般会", "一般都", "习惯", "老是这样", "老是", "老爱", "一向",
    "一贯", "永远", "永远都", "从来都", "基本上", "基本上都", "常常",
    "经常", "动不动", "素来", "每每", "历来",
)

# 频率断言：内容里出现它，就是在宣称「这是常态」。
FREQUENCY_MARKERS = (
    "每天", "天天", "总是", "一直", "每次", "每回", "以后", "永远", "从来",
    "从不", "老是", "每每", "一贯", "通常", "常常", "经常", "动不动",
    "无论何时", "不管什么时候",
)

# 绝对断言：内容里出现它，就是在宣称「永远如此」。
ABSOLUTE_MARKERS = (
    "永远", "绝不", "从来不", "从不", "任何时候", "任何情况下", "一律",
    "绝对", "注定", "根本不会", "完全不会", "决不",
)

# 长期性证据：原话里有它，才说明用户说的是「一贯如此」。
DURABILITY_MARKERS = (
    "喜欢", "讨厌", "反感", "不喜欢", "不吃", "不喝", "不爱", "最爱",
    "最喜欢", "最讨厌", "最不", "偏好", "爱好", "习惯", "通常", "一般会",
    "一般都", "总是", "每天", "天天", "一直", "每次", "以后", "从来",
    "永远", "绝不", "从不", "记住", "记下", "一向", "一贯", "爱吃",
    "爱喝", "爱看", "爱玩", "素来", "历来", "向来",
)

# 临时状态：眼下如此，不值得占着记忆库。
TRANSIENT_MARKERS = (
    "暂时", "先不", "暂时不", "暂时没", "发不了", "没法发", "发不出去",
    "被拦", "拦下", "拦下来", "拦截", "卡住", "卡了", "卡在", "挂了",
    "崩了", "在忙", "还没", "等一下", "等会儿", "一会儿", "回头",
    "正在", "待会", "待会儿", "这会儿", "来不及", "临时",
    "先欠着", "挂着", "故障", "报错", "失败",
)

# 临时情绪：谁都不记。
MOOD_MARKERS = (
    "好累", "很累", "太累", "累死", "好困", "很困", "太困", "困死",
    "困得", "好饿", "很饿", "饿死", "饿得", "好烦", "很烦", "烦死",
    "烦心", "心累", "好难过", "很难过", "难过", "伤心", "好伤心",
    "生气", "好气", "气死", "开心", "好开心", "高兴", "难受", "好难受",
    "想哭", "崩溃", "心态崩", "emo",
)

# 指令式表达：当次的吩咐，不是长期立场。
_DIRECTIVE_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"帮我",
        r"你来",
        r"你要(用|说|写|发|改成|换成|按|来)",
        r"你(用|说|写|发|改)",
        r"用你(的)?(语气|口吻|风格|人设)",
        r"语气.{0,4}(想要|要|改成|换成|用|换)",
        r"(改成|写成|换成)",
        r"这次(用|先|就|按|别)",
        r"先(这样|这么)",
        r"回头(再|再说)",
        r"按我(说的|要求)|照我(说的|要求)",
        r"我想要",
        r"这次先",
    )
)


STABLE_KINDS = frozenset(
    {MemoryKind.SEMANTIC, MemoryKind.PREFERENCE, MemoryKind.RELATION, MemoryKind.SKILL}
)


def _has(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _looks_directive(text: str) -> bool:
    """这句话像不像「当次的吩咐」。"""

    return any(pattern.search(text) for pattern in _DIRECTIVE_PATTERNS)


def _contains(phrase: str, haystack: str) -> float:
    """短句被长句覆盖的比例 ∈ [0, 1]。

    为什么不用 Dice：记忆是原话的改写（「今天只有一节课」被改成
    「季怡每天只有一节课」），两边字面重合得不多，Dice 只有 0.27；
    但宾语「一节课」是原封不动的，覆盖度就是 1.0——
    **这条记忆确实来自那句话**。
    """

    tp = token_set(phrase)
    if not tp:
        return 0.0
    th = token_set(haystack)
    if not th:
        return 0.0
    return len(tp & th) / len(tp)


# ---------------------------------------------------------------------- #
# 依据原话
# ---------------------------------------------------------------------- #

@dataclass
class Evidence:
    """一条候选记忆的出处。"""

    quote: str = ""
    """用户原话（截断过）。"""

    score: float = 0.0
    """匹配置信度 ∈ [0, 1]。"""

    window: Optional[Sequence[str]] = None
    """本次抽取窗口里用户说过的全部话（只留用户自己的）。"""

    def __init__(self, quote: str = "", score: float = 0.0, window: Optional[Sequence[str]] = None) -> None:
        self.quote = quote
        self.score = score
        self.window = list(window or ())


def _candidate_phrases(draft: MemoryDraft) -> list[str]:
    """用记忆的几个片段去原话里找它。"""

    phrases = [draft.content or ""]
    obj = (draft.object or "").strip()
    pred = (draft.predicate or "").strip()
    if obj:
        phrases.append(obj)
        if pred:
            phrases.append(pred + obj)
    subject = (draft.subject or "").strip()
    if subject and phrases:
        phrases.append(phrases[0].replace(subject, "", 1).strip())
    out: list[str] = []
    for phrase in phrases:
        phrase = (phrase or "").strip()
        if len(phrase) >= 2 and phrase not in out:
            out.append(phrase)
    return out


def find_evidence(
    draft: MemoryDraft,
    messages: Sequence[dict[str, Any]],
    max_chars: int = 120,
) -> Evidence:
    """给一条候选记忆找它的出处。

    只在**用户自己说的话**里找——机器人说过的话是上下文，
    不是用户的陈述，拿它当依据等于拿机器人的话当用户的证词。
    """

    user_lines: list[str] = []
    for msg in messages or ():
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").strip() == "assistant":
            continue
        content = str(msg.get("content") or msg.get("text") or "").strip()
        if content:
            user_lines.append(content)
    if not user_lines:
        return Evidence()

    phrases = _candidate_phrases(draft)
    best_text = ""
    best_score = 0.0
    for line in user_lines:
        score = 0.0
        for phrase in phrases:
            score = max(score, _contains(phrase, line), similarity(phrase, line))
        if score > best_score:
            best_score = score
            best_text = line
    if not best_text:
        return Evidence(window=user_lines)
    return Evidence(
        quote=best_text[:max_chars],
        score=best_score,
        window=user_lines,
    )


def attach_evidence(
    drafts: Sequence[MemoryDraft],
    messages: Sequence[dict[str, Any]],
    max_chars: int = 120,
) -> list[MemoryDraft]:
    """给候选记忆挂上依据原话（存进 meta）。"""

    out: list[MemoryDraft] = []
    for draft in drafts:
        meta = dict(draft.meta or {})
        evidence = find_evidence(draft, messages, max_chars=max_chars)
        if evidence.quote:
            meta["quote"] = evidence.quote
            meta["quote_score"] = round(evidence.score, 3)
        out.append(replace(draft, meta=meta))
    return out


# ---------------------------------------------------------------------- #
# 准入判断
# ---------------------------------------------------------------------- #

@dataclass
class Verdict:
    """一条候选记忆的判定结果。"""

    action: str = "keep"
    """keep / downgrade / drop"""

    reason: str = ""


def check_draft(draft: MemoryDraft, evidence: Optional[Evidence] = None) -> Verdict:
    """判断一条候选记忆能不能以「稳定事实」的形态入库。"""

    content = draft.content or ""
    if not content:
        return Verdict("drop", "空内容")

    if evidence is not None:
        quote = evidence.quote or ""
        window = list(evidence.window or ())
    else:
        quote = str((draft.meta or {}).get("quote") or "")
        window = []

    # 1) 临时情绪——谁都不记。
    if _has(content, MOOD_MARKERS):
        return Verdict("drop", "临时情绪不入库")

    # 2) 临时状态——眼下如此，不值得占着记忆库。
    if _has(content, TRANSIENT_MARKERS):
        return Verdict("drop", "临时状态不入库")

    if draft.kind not in STABLE_KINDS:
        return Verdict("keep")

    # 3) 时效范围：说「今天/这次/刚才」的事，不许当成稳定事实。
    if _has(content, TIME_MARKERS) and not _has(content, PERMANENCE_MARKERS):
        return Verdict("downgrade", "一次性情况被当成了稳定事实")

    # 4) 断言与依据对不上。
    #    「每天/总是/永远」是**常态断言**：依据里必须有同样的说法。
    #    依据里出现的是「今天/整天/这次」，那更糟——把一次性升级成了常态。
    if _has(content, FREQUENCY_MARKERS) or _has(content, ABSOLUTE_MARKERS):
        if quote and _has(quote, TIME_MARKERS):
            return Verdict("downgrade", "原话说的是这一次，记忆却写成了常态")
        if quote and not _has(quote, FREQUENCY_MARKERS + ABSOLUTE_MARKERS):
            window_support = any(
                _has(line, FREQUENCY_MARKERS + ABSOLUTE_MARKERS) for line in window
            )
            if not window_support:
                return Verdict("downgrade", "原话里没有这个频率，是模型自己加的")

    # 5) 当次的吩咐 ≠ 长期偏好。
    #    「这次你用平时跟别人说话的语气」是一次性要求；
    #    「我喜欢这种语气」才是长期偏好。
    if draft.kind is MemoryKind.PREFERENCE:
        source = quote or " ".join(window)
        if source and not _has(source, DURABILITY_MARKERS) and _looks_directive(source):
            return Verdict("downgrade", "这是一次性的吩咐，不是长期偏好")

    return Verdict("keep")


def guard_drafts(
    drafts: Sequence[MemoryDraft],
    messages: Sequence[dict[str, Any]] = (),
    store_evidence: bool = True,
    max_chars: int = 120,
) -> tuple[list[MemoryDraft], list[tuple[MemoryDraft, str]]]:
    """跑一遍闸门。

    返回 ``(留下的, [(被丢掉的, 原因)])``。

    被降级的记忆会改成 episodic——**信息不丢，但不再常驻注入**，
    交给淡化引擎自然淘汰。
    """

    kept: list[MemoryDraft] = []
    dropped: list[tuple[MemoryDraft, str]] = []
    for draft in drafts:
        if messages:
            evidence = find_evidence(draft, messages, max_chars=max_chars)
        else:
            evidence = Evidence(
                quote=str((draft.meta or {}).get("quote") or ""),
                score=float((draft.meta or {}).get("quote_score") or 0.0),
            )
        verdict = check_draft(draft, evidence=evidence)
        if verdict.action == "drop":
            dropped.append((draft, verdict.reason))
            continue
        if verdict.action == "downgrade":
            meta = dict(draft.meta or {})
            meta["downgraded_from"] = draft.kind.value
            meta["downgrade_reason"] = verdict.reason
            draft = replace(draft, kind=MemoryKind.EPISODIC, meta=meta)
            dropped.append((draft, verdict.reason))
        if store_evidence and evidence.quote:
            meta = dict(draft.meta or {})
            meta.setdefault("quote", evidence.quote)
            meta.setdefault("quote_score", round(evidence.score, 3))
            draft = replace(draft, meta=meta)
        kept.append(draft)
    return kept, dropped


def check_remember_request(
    draft: MemoryDraft,
    *,
    focus_subjects: Sequence[str] = (),
    self_names: Sequence[str] = (),
) -> Verdict:
    """判断「有人明确要求记住」这条请求能不能入库。

    与抽取路径的区别：抽取出来的候选是模型**自作主张**的，
    这里的条目是**有人点名要记**的——所以只拦两类：

    1. **临时状态与临时情绪**：记了也没用（「暂时发不了」「好累」）；
    2. **不是关于该记的那个人**：主语不是、内容里也没提到——
       这条跟「要记的人」无关，多半是模型把别处的信息塞进来了。

    注意第 2 条的判定：**主语是机器人自己不一定该丢**。
    「许杏玖欠季怡半首歌」这种**承诺**是该记的——
    它虽然主语是机器人，说的却是关于季怡的事。
    所以判据是「跟那个人有没有关系」，不是「主语是不是那个人」。

    为什么这条闸门必须存在（实测漏过一条）：
    模型调用「记住」工具时，工具路径**不经过抽取那一整套纪律**，
    于是「不吃肥肠有0+0姬子启行、0+0星期日……」这种游戏数据
    被当成「关于某个人的稳定事实」记了进去（主语还是群里另一个人）。
    """

    content = draft.content or ""
    if not content.strip():
        return Verdict("drop", "内容是空的")
    if _has(content, MOOD_MARKERS):
        return Verdict("drop", "临时情绪不入库")
    if _has(content, TRANSIENT_MARKERS):
        return Verdict("drop", "临时状态不入库")

    subject = (draft.subject or "").strip()
    focus_set = {str(s).strip() for s in focus_subjects if str(s).strip()}
    if focus_set:
        about_focus = any(name and name in content for name in focus_set)
        if subject in focus_set:
            about_focus = True
        if not about_focus:
            return Verdict("drop", "跟要记的那个人没关系")
    elif self_names:
        # 没配 focus_subjects 时，至少别记机器人自己的事。
        self_set = {str(s).strip() for s in self_names if str(s).strip()}
        if subject and subject in self_set:
            return Verdict("drop", "机器人自己的事不入库")
    return Verdict("keep")
