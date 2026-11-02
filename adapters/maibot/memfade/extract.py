"""记忆抽取。

两条路：

1. **LLM 抽取**（主）：一次给一批消息，让模型输出结构化的记忆候选。
2. **规则抽取**（兜底）：模型不可用、限流、超时的时候，用正则抓最要命的
   几类信息（姓名、偏好、承诺、称呼）。宁可少记，不可乱记。

## 抽取纪律

提示词里钉死了几条硬规则，它们对应真实踩过的坑：

* **一次性事件不当长期事实**。上次买了《影之刃零》、昨天吵了一架——
  这些是 episodic，显著性压低，靠淡化引擎自然退场，而不是每次聊天都注入。
* **同一件事只写一次**。避免「四个赞」被写成四条。
* **用户明确说「记住」才允许跨会话共享**。默认一律会话内可见。
* **不记密钥、密码、证件号**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

from .config import ExtractionConfig
from .llm import LLMClient, parse_json_block
from .textutil import split_atomic
from .types import MemoryDraft, MemoryKind


EXTRACTION_SYSTEM_PROMPT = (
    "你是一个聊天机器人的长期记忆抽取器。\n"
    "\n"
    "从对话中挑出**关于对话里那个人**、对未来聊天仍然有用的信息，"
    "输出 JSON 数组。不要输出任何解释文字。\n"
    "\n"
    "每条记忆的字段：\n"
    "- content: 一句完整、自洽的中文陈述（主语明确，不要写「他说」这种指代不明的话）\n"
    "- kind: episodic（发生过的具体事件）/ semantic（稳定事实）/ "
    "preference（喜好厌恶）/ relation（人与人关系）/ skill（习惯做法）\n"
    "- importance: 0~1 的小数。稳定事实 0.6~0.9；一次性事件 0.2~0.45；"
    "用户强调的 0.9 以上\n"
    "- subject / predicate / object: 把事实拆成三段，**必须填**。\n"
    "  subject 是「谁」@SUBJECT_HINT@，predicate 是「怎么样」（住在 / 喜欢 / 参加），\n"
    "  object 是「什么」（上海 / 辣的食物 / 第18届蓝桥杯）。\n"
    "  这三段是矛盾检测的唯一依据——「他住在北京」后来变成「他住在上海」，\n"
    "  靠它们才能发现旧值该退场。填不上就说明这条不值得记。\n"
    "- cross_scope: 只有当用户明确要求「记住这个 / 永远别忘」时才为 true，否则一律 false\n"
    "\n"
    "硬性要求：\n"
    "1. 只抽**用户本人**的事。以下一律不抽：\n"
    "   - 机器人（对方）自己的事——它看过什么、说过什么、收藏了什么，\n"
    "     那是它的人设，不是用户的记忆；\n"
    "   - 群里的公共信息、转发出来的公告、别人发的东西；\n"
    "   - 作品设定、剧情、赛事规则、播出时间、游戏机制这类公共知识。\n"
    "2. 引用、转发、复述里的内容不是用户的话，别记。"
    "「我把你那句总结存下来了」里的正文是别人写的。\n"
    "3. 用户否认、纠正、反问、收回的话不要记。"
    "「不是我看的是你看」只说明是个误会，别把它记成事实。\n"
    "4. 用户只是寒暄、提问、开玩笑、试探（问「你是谁」「你会骂人」这种）时，"
    "一条都不要抽，输出 []。\n"
    "5. **一条记忆只讲一件事**，30 字以内。\n"
    "   同一段话里有多件事就拆成多条分别输出。\n"
    "   反例（错）：「季怡在备赛蓝桥杯：报名只报名不缴费，先校赛再定省赛，"
    "10月16日有大纲解析直播」\n"
    "   正例（对）：「季怡参加蓝桥杯只报名不缴费」\n"
    "6. 一次性事件（临时状态、刚发生的小事、上次买了什么）可以抽，"
    "但必须标记为 episodic 且 importance 不超过 0.45。\n"
    "7. 同一件事只写一条，不要换措辞重复。\n"
    "8. 主语必须是**人**：用对话里出现过的名字，"
    "不要用「用户」「对方」「他」「她」，也不要拿作品名、节目名、赛事名当主语。\n"
    "9. 绝不记录密码、密钥、身份证号、银行卡号、验证码等敏感信息。\n"
    "10. 不确定就不抽。宁可少抽，不要抽错。\n"
    "11. 输出格式示例："
    '[{{"content":"季怡喜欢辣的食物","kind":"semantic","importance":0.8,'
    '"subject":"季怡","predicate":"喜欢","object":"辣的食物","cross_scope":false}}]'
)


EXTRACTION_USER_TEMPLATE = (
    "以下是最近的一段对话。请从中抽取值得长期记住的信息。\n"
    "\n"
    "<对话>\n{conversation}\n</对话>\n"
    "\n"
    "按 JSON 数组输出（没有值得记的就输出 []）："
)


def build_extraction_prompt(config: Optional["ExtractionConfig"] = None) -> str:
    """按配置拼抽取提示词。

    核心库不绑定任何人名：这次对话里有谁，由部署方通过
    ``subject_hint`` 填进来（形如「（许杏玖 / 季怡）」）。
    """

    hint = ""
    if config is not None and config.subject_hint:
        hint = config.subject_hint.strip()
    # 用 @...@ 而不是 {...}：提示词常量里出现字面花括号会让 .format() 崩，
    # 「所有提示词都能被 format」是仓库里的一条守卫测试。
    return EXTRACTION_SYSTEM_PROMPT.replace("@SUBJECT_HINT@", hint)


# 泛指词：主语写成这些，说明模型没真的认出「是谁」，
# 这条记忆没法做矛盾检测（subject 是空的或泛的），
# 也不能拿来生成画像——一律不算数。
GENERIC_SUBJECTS = frozenset(
    {
        "用户",
        "对方",
        "他",
        "她",
        "它",
        "我",
        "你",
        "我们",
        "你们",
        "他们",
        "咱",
        "本人",
        "自己",
        "人家",
    }
)

# 常见的「人」称身份词。这些不是说话人，但也确实是人，
# 主语是它们时该保留（「季怡的妈妈叫小红」这种关系）。
PERSON_ROLES = frozenset(
    {
        "妈妈",
        "爸爸",
        "母亲",
        "父亲",
        "哥哥",
        "姐姐",
        "弟弟",
        "妹妹",
        "儿子",
        "女儿",
        "爷爷",
        "奶奶",
        "外公",
        "外婆",
        "叔叔",
        "阿姨",
        "老师",
        "同学",
        "朋友",
        "室友",
        "同事",
        "老板",
        "领导",
        "师父",
        "师傅",
        "徒弟",
        "对象",
        "男朋友",
        "女朋友",
        "丈夫",
        "妻子",
        "医生",
        "客服",
        "房东",
        "邻居",
    }
)


def filter_drafts(
    drafts: Sequence[MemoryDraft],
    *,
    self_names: Sequence[str] = (),
    focus_subjects: Sequence[str] = (),
    participants: Sequence[str] = (),
    require_person: bool = True,
    default_subject: str = "",
) -> tuple[list[MemoryDraft], list[tuple[MemoryDraft, str]]]:
    """按「主语纪律」筛掉不该入库的候选。

    返回 ``(留下的, [(被丢掉的, 原因)])``。

    这道闸门是**代码**而不是提示词——提示词说「别记自己的事」，
    模型偶尔还是会犯；犯一次，用户的记忆库里就多一条
    「机器人自己看了什么番」，主人一眼就看出来被污染了。

    丢弃规则：

    1. 主语是机器人自己 → 丢（记忆库是「关于对方」的）；
    2. 主语是泛指词（用户/对方/他/她）→ 换成默认主语，或丢；
    3. 设了 ``focus_subjects`` 而主语不在其中 → 丢；
    4. 主语既不是本次对话的说话人、也不是常见身份词 → 丢
       （作品名、赛事名、链接名这类公共知识不是「关于某个人」的记忆）。
    """

    self_set = {str(n).strip() for n in self_names if str(n).strip()}
    focus_set = {str(n).strip() for n in focus_subjects if str(n).strip()}
    people_set = {str(n).strip() for n in participants if str(n).strip()}
    fallback = str(default_subject or "").strip()

    kept: list[MemoryDraft] = []
    dropped: list[tuple[MemoryDraft, str]] = []
    for draft in drafts:
        subject = (draft.subject or "").strip()
        if subject and subject in self_set:
            dropped.append((draft, "self_fact"))
            continue
        if not subject:
            # 模型没填主语：能确定是对话里哪个人就替他填上，
            # 确定不了（有多个说话人且没配 focus）就丢掉——
            # 提示词里写着「填不上就说明这条不值得记」。
            if fallback and (not focus_set or fallback in focus_set):
                draft = replace(draft, subject=fallback)
            elif people_set:
                dropped.append((draft, "no_subject"))
                continue
        elif subject in GENERIC_SUBJECTS:
            if fallback and (not focus_set or fallback in focus_set):
                draft = replace(draft, subject=fallback)
            elif people_set:
                # 知道这次对话里有谁，模型却还写「用户/对方」——
                # 说明它没认出主语，这条没法用来做矛盾检测，丢掉。
                dropped.append((draft, "generic_subject"))
                continue
            # 连说话人都不知道（老宿主没有昵称字段），就先留着，
            # 总比把真事实扔掉强。
        else:
            if focus_set and subject not in focus_set:
                dropped.append((draft, "not_focus"))
                continue
            if (
                require_person
                and people_set
                and subject not in people_set
                and subject not in PERSON_ROLES
                and subject not in focus_set
            ):
                dropped.append((draft, "not_a_person"))
                continue
        kept.append(draft)
    return kept, dropped

GIST_SYSTEM_PROMPT = (
    "你把一条记忆压缩成一行不超过 {max_chars} 个字的摘要，"
    "保留关键信息（人、事、物、数字），去掉寒暄与语气词。"
    "**只做压缩，不要补充原文里没有的信息**，"
    "尤其不要自己加日期、集数、版本号这类看起来该有的细节。"
    "只输出摘要本身，不要解释。"
)


GIST_USER_TEMPLATE = "记忆原文：\n{content}\n\n压缩成一行摘要："


CANONICALIZE_PROMPT = (
    "把下面这些关于人的零散事实，规范化成「一件事一条」的列表。\n"
    "\n"
    "规则：\n"
    "1. 一条只讲一件事，30 字以内；同一件事的多种说法合并成一条。\n"
    "2. 给出 subject（谁）、predicate（怎么样）、object（什么）三段，必须填。\n"
    "   predicate 只是一个**动宾短语**，6 字以内（如「住在」「喜欢」「参加」「称呼为」），\n"
    "   不要把整句话塞进 predicate。\n"
    "3. 说法不同但意思一样的，输出成同一条（内容取最完整的那版）。\n"
    "4. 意思不同的（比如称呼一个是「宝宝」一个是「Master」），**分成两条**，不要合并。\n"
    "5. 保留原文的人名，不要换成别的称呼。\n"
    "6. **只做拆分与合并，绝不补充原文里没有的信息**——"
    "不要加集数、日期、版本号、书名号里的作品名。"
    "原文没说是第几季，就不要写「第五季」。\n"
    "\n"
    "输出 JSON 数组，不要解释：\n"
    '[{{"content":"...","subject":"...","predicate":"...","object":"..."}}]\n'
    "\n"
    "待规范化的事实：\n"
    "{facts}"
)

CANONICALIZE_SPLIT_PROMPT = (
    "把下面每一条长记忆拆成「一件事一条」。\n"
    "\n"
    "规则：\n"
    "1. **每条输入都要有输出**，一条都不能漏。\n"
    "2. 拆出来的每条只讲一件事，30 字以内。\n"
    "3. 给出 subject（谁）、predicate（怎么样）、object（什么）三段，必须填。\n"
    "   predicate 只是一个**动宾短语**，6 字以内（如「参加」「住在」「喜欢」），\n"
    "   不要把整句话塞进 predicate。\n"
    "4. 保留原文的人名和具体数字，不要改写。\n"
    "5. 原文本来就只有一件事的，就原样输出，三元组照填。\n"
    "6. **只做拆分，不要补充原文里没有的信息**——"
    "不要加集数、日期、版本号、作品名。原文没说是第几季就别写第几季。\n"
    "\n"
    "输出 JSON **对象**：键是输入的编号（从 0 开始），值是这一条拆出来的列表。\n"
    "不要解释，不要 markdown 代码块：\n"
    '{{"0": [{{"content":"...","subject":"...","predicate":"...","object":"..."}}], "1": []}}\n'
    "\n"
    "待拆解的记忆：\n"
    "{facts}"
)

TRIPLE_PROMPT = (
    "给下面每一条事实补上三元组 (谁, 怎么样, 什么)。\n"
    "\n"
    "规则：\n"
    "1. subject 填主语（人名或事件名），predicate 填动宾短语（6 字以内），\n"
    "   object 填宾语。\n"
    "2. 一条对应一条输出，顺序一致，不要合并、不要拆分。\n"
    "3. **只从原文里取**，不要补充原文没有的信息。\n"
    "4. 只输出 JSON 数组，不要解释。\n"
    "\n"
    "事实：\n"
    "{facts}\n"
    '\n输出：[{{"index":0,"subject":"...","predicate":"...","object":"..."}}]'
)


@dataclass
class ExtractionResult:
    drafts: list[MemoryDraft] = field(default_factory=list)
    raw: str = ""
    used_llm: bool = False
    error: str = ""
    dropped: list[tuple[str, str]] = field(default_factory=list)
    """被「主语纪律」丢掉的候选：(内容, 原因)。给宿主看日志用。"""


_KIND_ALIASES = {
    "episodic": MemoryKind.EPISODIC,
    "episode": MemoryKind.EPISODIC,
    "event": MemoryKind.EPISODIC,
    "semantic": MemoryKind.SEMANTIC,
    "fact": MemoryKind.SEMANTIC,
    "preference": MemoryKind.PREFERENCE,
    "relation": MemoryKind.RELATION,
    "skill": MemoryKind.SKILL,
}

# 各档显著性的合法区间。
# 提示词里已经约束过一遍，但**提示词不是保证**——模型偶尔会给一次性事件打 0.9，
# 那会让「上周吃了炸鸡」比「他叫许杏玖」活得久、权重还更高。
# 所以在代码里再夹一次。
IMPORTANCE_BOUNDS = {
    MemoryKind.EPISODIC: (0.15, 0.55),
    MemoryKind.SEMANTIC: (0.40, 0.95),
    MemoryKind.PREFERENCE: (0.40, 0.90),
    MemoryKind.RELATION: (0.40, 0.95),
    MemoryKind.SKILL: (0.40, 0.90),
}


def _clamp_importance(kind: MemoryKind, value: float) -> float:
    """按种类夹紧显著性。"""

    low, high = IMPORTANCE_BOUNDS.get(kind, (0.0, 1.0))
    return min(high, max(low, float(value)))


def _is_rate_limited(exc: BaseException) -> bool:
    """这个异常是不是「限流」？

    限流（429 / RateLimitError / 「请求过于频繁」）不是「这次太大」，
    重试只会继续撞墙、继续烧额度——该立刻退回规则抽取。
    """

    text = "%s %s" % (type(exc).__name__, exc)
    markers = ("429", "ratelimit", "rate_limit", "too many requests", "请求过于频繁")
    return any(marker in text.lower() for marker in markers)


class MemoryExtractor:
    """把对话变成候选记忆。"""

    def __init__(self, config: Optional[ExtractionConfig] = None, llm: Optional[LLMClient] = None) -> None:
        self.config = config or ExtractionConfig()
        self.llm = llm

    # ------------------------------------------------------------------ #
    # 公开入口
    # ------------------------------------------------------------------ #

    async def extract(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        source: str = "",
    ) -> ExtractionResult:
        """从消息列表抽取候选记忆。"""

        conversation = self._render(messages)
        if not conversation:
            return ExtractionResult()

        if self.llm is not None:
            try:
                raw, drafts = await self._complete_with_retry(
                    EXTRACTION_USER_TEMPLATE.format(conversation=conversation),
                    system=build_extraction_prompt(self.config),
                    max_tokens=4000,
                )
                kept, dropped = self._apply_subject_policy(drafts, messages)
                return ExtractionResult(
                    drafts=kept,
                    raw=raw,
                    used_llm=True,
                    dropped=[(d.content, reason) for d, reason in dropped],
                )
            except Exception as exc:
                fallback = self._rule_extract(conversation) if self.config.use_rules_fallback else []
                kept, dropped = self._apply_subject_policy(fallback, messages)
                return ExtractionResult(
                    drafts=kept,
                    used_llm=False,
                    error=str(exc),
                    dropped=[(d.content, reason) for d, reason in dropped],
                )

        if self.config.use_rules_fallback:
            kept, dropped = self._apply_subject_policy(self._rule_extract(conversation), messages)
            return ExtractionResult(
                drafts=kept,
                used_llm=False,
                dropped=[(d.content, reason) for d, reason in dropped],
            )
        return ExtractionResult()

    def _apply_subject_policy(
        self,
        drafts: Sequence[MemoryDraft],
        messages: Sequence[dict[str, Any]],
    ) -> tuple[list[MemoryDraft], list[tuple[MemoryDraft, str]]]:
        """按「主语纪律」筛一遍抽取结果。"""

        if not drafts:
            return [], []
        participants = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            name = str(msg.get("speaker") or "").strip()
            if name and name not in participants:
                participants.append(name)
        default_subject = ""
        focus = [n.strip() for n in self.config.focus_subjects if n and n.strip()]
        if focus:
            default_subject = focus[0]
        elif len(participants) == 1:
            default_subject = participants[0]
        kept, dropped = filter_drafts(
            drafts,
            self_names=self.config.self_names,
            focus_subjects=self.config.focus_subjects,
            participants=participants,
            require_person=self.config.require_person_subject,
            default_subject=default_subject,
        )
        return kept, dropped

    async def summarize(self, content: str, *, max_chars: int = 40) -> str:
        """把一条记忆压成 gist 摘要。LLM 不可用时退回规则兜底。"""

        text = (content or "").strip()
        if not text:
            return ""
        if self.llm is not None:
            try:
                raw = await self.llm.complete(
                    GIST_USER_TEMPLATE.format(content=text),
                    system=GIST_SYSTEM_PROMPT.format(max_chars=max_chars),
                    max_tokens=120,
                    temperature=0.1,
                )
                lines = (raw or "").strip().splitlines()
                if lines and lines[0]:
                    return lines[0][:max_chars]
            except Exception:
                pass
        from .textutil import best_clause

        return best_clause(text, max_chars)

    async def canonicalize(self, facts: Sequence[str]) -> list[dict]:
        """把零散事实规范化成「一件事一条 + 带三元组」。

        供迁移旧记忆用：旧库里同一件事常有三四种说法，
        先让模型合并成一条并给出 (谁,怎么样,什么)，
        去重和矛盾检测才有抓手。
        """

        if not self.llm is None and not facts:
            return []
        if self.llm is None:
            return [{"content": f, "subject": "", "predicate": "", "object": ""} for f in facts]
        prompt = CANONICALIZE_PROMPT.format(facts="\n".join(f"- {f}" for f in facts))
        raw, items = await self._complete_with_retry(prompt, max_tokens=8000)
        payload = parse_json_block(raw)
        items = payload if isinstance(payload, list) else ([payload] if payload else [])
        out: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            out.append(
                {
                    "content": content,
                    "subject": str(item.get("subject") or "").strip(),
                    "predicate": str(item.get("predicate") or "").strip(),
                    "object": str(item.get("object") or "").strip(),
                }
            )
        if not out:
            # 模型没给出可解析的结果（可能被 max_tokens 截断）
            raise RuntimeError("规范化失败：模型没有返回可解析的 JSON")
        return out

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #

    def _render(self, messages: Sequence[dict[str, Any]]) -> str:
        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip()
            content = str(msg.get("content") or msg.get("text") or "").strip()
            if not content:
                continue
            speaker = str(msg.get("speaker") or "").strip()
            if role == "user":
                lines.append(f"{speaker or '用户'}：{content}")
            elif role == "assistant":
                lines.append(f"{speaker or '机器人'}：{content}")
            else:
                lines.append(f"{speaker or role or '对话'}：{content}")
        return "\n".join(lines)

    async def _complete_with_retry(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 4000,
        attempts: int = 3,
    ) -> tuple:
        """调用 LLM 并保证拿到可解析的结果。

        为什么要重试：有些模型（MiniMax 这类默认开着 thinking 的）
        会把 token 全烧在思考上，最后 finish_reason=length、输出为空。
        宿主日志会打「达到最大 token 限制…无法生成完整内容」的警告，
        但对插件来说就是「什么都没抽到」。
        这里的做法是：**空输出就加大预算重来**，而不是默默记 0 条。

        ⚠ **限流（429）不重试**：限流不是「这次太大」，重试只会
        继续撞墙、继续烧额度，而且把退路（规则抽取）拖慢了。
        撞上就立刻退回规则抽取。
        """

        raw = ""
        budget = max_tokens
        last_error = ""
        for _ in range(max(1, attempts)):
            try:
                raw = await self.llm.complete(
                    prompt,
                    system=system,
                    max_tokens=budget,
                    temperature=0.1,
                )
            except Exception as exc:
                last_error = str(exc)
                if _is_rate_limited(exc):
                    raise RuntimeError(f"抽取失败（模型限流）：{last_error}") from exc
                # 超时/网络错误：说明这次**太大**了，减小预算再来，
                # 而不是加大（越大越容易超时）。
                budget = max(1000, budget // 2)
                continue
            drafts = self._parse(raw)
            if drafts:
                return raw, drafts
            # 输出为空：是被 max_tokens 截断了，加大预算重来
            last_error = "模型没有返回可解析的结果"
            budget *= 2
        raise RuntimeError(f"抽取失败：{last_error or '模型无输出'}")

    def _parse(self, raw: str) -> list[MemoryDraft]:
        """解析 LLM 输出。脏数据一律丢弃，不猜。"""

        payload = parse_json_block(raw)
        if payload is None:
            return []
        if isinstance(payload, dict):
            payload = payload.get("memories") or payload.get("items") or [payload]
        if not isinstance(payload, list):
            return []

        drafts: list[MemoryDraft] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content or len(content) > 500:
                continue
            if self._looks_sensitive(content):
                continue
            kind = _KIND_ALIASES.get(str(item.get("kind") or "").lower(), MemoryKind.EPISODIC)
            importance = _clamp_importance(kind, _clamp_float(item.get("importance"), 0.5))
            cross = bool(item.get("cross_scope")) and self.config.promote_cross_scope
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            object_value = str(item.get("object") or "").strip()
            source = str(item.get("source") or "").strip()
            # ⚠ 提示词说了「30 字以内」不算数——模型偶尔会甩一整段回来。
            # 超长的按标点拆开，一条变成几条，别让长记忆进库。
            pieces = split_atomic(content, max_chars=self.config.max_chars)
            if len(pieces) <= 1:
                pieces = [content]
            for index, piece in enumerate(pieces):
                drafts.append(
                    MemoryDraft(
                        content=piece,
                        kind=kind,
                        importance=importance,
                        # 三元组只挂第一条：它是「主语 + 第一件事」，
                        # 后面的碎片脱离了这个三元组。
                        subject=subject if index == 0 else "",
                        predicate=predicate if index == 0 else "",
                        object=object_value if index == 0 else "",
                        source=source,
                        cross_scope=cross,
                    )
                )
        drafts.sort(key=lambda d: d.importance, reverse=True)
        return drafts[: self.config.max_per_batch]

    def _looks_sensitive(self, content: str) -> bool:
        """敏感内容不进记忆库。"""

        patterns = (
            r"密码",
            r"密钥",
            r"验证码",
            r"身份证号?\s*[:：]?\s*\d{6,}",
            r"银行卡号?\s*[:：]?\s*\d{8,}",
            r"\bsk-[A-Za-z0-9]{16,}",
            r"api[_ ]?key\s*[:=]",
        )
        return any(re.search(p, content, re.IGNORECASE) for p in patterns)

    # -- 规则兜底 ------------------------------------------------------- #

    _RULES: tuple[tuple[str, MemoryKind, float, bool], ...] = (
        (r"(?:记住|记一下|永远记住|别忘了)[：:，, ]*(.+)", MemoryKind.SEMANTIC, 0.95, True),
        (r"(?:我|他|她)(?:叫|的名字是|名字叫)\s*([^\s，。！？,.!?]{1,12})", MemoryKind.SEMANTIC, 0.9, False),
        (r"(?:我|他|她)(?:最喜欢|超喜欢|超爱)\s*([^，。！？,.!?]{1,24})", MemoryKind.PREFERENCE, 0.8, False),
        # 光说「我喜欢X」的写法在中文里最常见，早先的规则只认「最喜欢」，
        # 于是模型一限流（429）就只剩下「不叫/不讨厌」那几条能抽到。
        # 「我最喜欢X」不会被下面这条重复命中（「喜欢」前面是「最」，对不上）。
        (r"(?:我|他|她)喜欢\s*([^，。！？,.!?]{1,24})", MemoryKind.PREFERENCE, 0.75, False),
        (r"(?:我|他|她)(?:最讨厌|受不了|根本不吃|不喜欢|讨厌|不吃)\s*([^，。！？,.!?]{1,24})", MemoryKind.PREFERENCE, 0.75, False),
        (r"(?:我|他|她)(?:住在|家在)\s*([^，。！？,.!?]{1,24})", MemoryKind.SEMANTIC, 0.72, False),
        (r"(?:我|他|她)(?:的)?(?:是做|工作是|职业是)\s*([^，。！？,.!?]{1,24})", MemoryKind.SEMANTIC, 0.72, False),
        (r"(?:约好|说好|约定|答应)\s*([^。！？.!?]{1,32})", MemoryKind.EPISODIC, 0.68, False),
    )

    def _rule_extract(self, conversation: str) -> list[MemoryDraft]:
        """正则兜底抽取。"""

        from .textutil import normalized_hash

        drafts: list[MemoryDraft] = []
        seen: set[str] = set()
        for pattern, kind, importance, cross in self._RULES:
            for match in re.finditer(pattern, conversation):
                tail = match.group(1).strip()
                if not tail:
                    continue
                # 整句留着（「我住在上海」比「上海」完整），
                # 只有「记住：xxx」这种前缀是噪音，要去掉。
                content = match.group(0).strip()
                if content.startswith(("记住", "记一下", "永远记住", "别忘了")):
                    content = tail
                if len(content) < 2 or len(content) > 120:
                    continue
                key = normalized_hash(content)
                if key in seen:
                    continue
                seen.add(key)
                drafts.append(
                    MemoryDraft(
                        content=content,
                        kind=kind,
                        importance=importance,
                        cross_scope=cross and self.config.promote_cross_scope,
                        source="rule",
                    )
                )
        drafts.sort(key=lambda d: d.importance, reverse=True)
        return drafts[: self.config.max_per_batch]


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))

