"""中文友好的文本处理。

记忆检索里的中文有个老问题：SQLite FTS5 自带的 unicode61 分词器会把一整句
中文切成「一个词」，导致「炸鸡」搜不到「今天吃了炸鸡」。

这里不引入分词依赖，而是用**字符二元组**（character bigram）做索引：

    今天吃了炸鸡  ->  今天 天吃 吃了 了炸 炸鸡

二元组天然适配中文，也不需要词典。英文/数字则额外保留整词，
因为 bigram 对拉丁字母的召回质量明显不如整词。

这套做法借鉴自 memory-brain（AstrBot 记忆大脑）的 FTS5 实现。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

_PUNCT_RE = re.compile(r"[\s　!-/:-@\[-`{-~ -⁯、-〿！-･]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def normalize(text: str) -> str:
    """NFKC 规范化 + 折叠空白。保留标点，用于展示。"""

    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", folded).strip()


def normalize_light(text: str) -> str:
    """只折叠空白，**不动标点**。

    NFKC 会把全角标点折成半角：「，」→「,」、「（」→「(」。
    那是给**索引**用的（查询和内容两边都折，匹配才一致）；
    但**给人看的**内容（gist 摘要）折了就变味了——
    一段中文里混着半角逗号，读着就不是那个味。
    """

    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(text: str) -> str:
    """归一化成去标点、去空白、小写的比较键。

    用于精确去重：「二十块加价仲裁。」和「二十块加价仲裁」应当是同一条。
    """

    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return _PUNCT_RE.sub("", folded)


def normalized_hash(text: str) -> str:
    """归一化内容的稳定哈希。"""

    return hashlib.sha1(normalize_key(text).encode("utf-8")).hexdigest()


def bigrams(text: str) -> list[str]:
    """字符二元组。中文为主，英文按词补充。"""

    key = normalize(text)
    if not key:
        return []

    out: list[str] = []
    cjk_runs = _CJK_RE.findall(key)
    # 按连续 CJK 片段切二元组，避免跨标点拼出无意义的组合
    for run in _CJK_RUN_RE.findall(key):
        if len(run) == 1:
            out.append(run)
        else:
            for i in range(len(run) - 1):
                out.append(run[i : i + 2])

    out.extend(word.lower() for word in _WORD_RE.findall(key))
    return out


def cjk_unigrams(text: str) -> list[str]:
    """单个汉字，去重但保持出现顺序。"""

    key = normalize(text)
    if not key:
        return []
    return list(dict.fromkeys(_CJK_RE.findall(key)))


def content_chars(text: str) -> list[str]:
    """单个实义汉字（去掉「的」「了」「是」这类虚词）。

    单字兜底通道用这个：拿「的」去检索等于跟整本语料库匹配。
    """

    return [c for c in cjk_unigrams(text) if c not in _FUNCTION_CHARS]


def fts_text(text: str) -> str:
    """喂给 FTS5 的索引串：二元组之间用空格隔开。

    ⚠ 这里**只存二元组**，不要把单字也塞进来。
    同一个 FTS 表里混着单字，BM25 的长度归一化会被带偏：
    长文档涨的 token 比短文档多，长记忆被压得更狠，
    「重要的长约定」会被「炸鸡」这种短条目反超。
    单字查询另有专门的通道（``store.search_substring``）。
    """

    return " ".join(bigrams(text))


def token_set(text: str) -> set[str]:
    """相似度比较用的 token 集合。"""

    return set(bigrams(text))


def similarity(a: str, b: str) -> float:
    """二元组 Dice 系数 ∈ [0, 1]。

    为什么不用包含度（交集 / 较短者）：包含度在「短句完全是长句的子集」
    时会给出 1.0，于是「二十块加价仲裁还没解决」和
    「二十块加价仲裁还没解决呢」会被判成完全相同——多一个语气词而已。

    为什么不用 Jaccard：记忆摘要常常是原文的缩写，两者长度天然不等，
    Jaccard 对这种长度差过于敏感。

    Dice 兼顾两头：2|A∩B| / (|A|+|B|)。
    """

    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if not inter:
        return 0.0
    return (2.0 * inter) / (len(ta) + len(tb))


def jaccard(a: str, b: str) -> float:
    """二元组 Jaccard 相似度 ∈ [0, 1]。"""

    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def estimate_tokens(text: str) -> int:
    """粗略 token 估算。中文约 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token。"""

    if not text:
        return 0
    key = normalize(text)
    cjk = len(_CJK_RE.findall(key))
    rest = len(_CJK_RE.sub("", key))
    return int(cjk + rest / 4.0) + 1


def split_sentences(text: str) -> list[str]:
    """中英混排的句子切分。"""

    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(normalize(text))]
    return [p for p in parts if p]


def truncate(text: str, limit: int, ellipsis: str = "…") -> str:
    """按字符截断。"""

    # 用轻量规范化：截断出来的内容是要给人看的，
    # NFKC 把全角标点折成半角就变味了。
    key = normalize_light(text)
    if limit <= 0 or len(key) <= limit:
        return key
    return key[: max(0, limit - len(ellipsis))] + ellipsis




def best_clause(text: str, limit: int = 40) -> str:
    """取一句话作为 gist（内容本来就短就整段留）。

    为什么是第一句：gist 的作用是「一件事一行」，
    而**第一句通常就是那件事**（后面的是补充、后续、别的事）。

    早先试过「挑最有信息量的一句」（按内容词数量打分），结果更糟：
    「后来他跟朋友去吃了炸鸡，许杏玖很开心，然后又去买了点东西…」
    会挑到最后那句——正好把「发生了什么」挑没了。
    第一句简单、可预测、留着话题。

    ⚠ 分句点不含「冒号后面跟数字」的情况：「10月16日19:00的直播」
    不能被劈成「10月16日19」。
    """

    key = normalize_light(text)
    if not key:
        return ""
    if len(key) <= limit:
        # 本来就够短，压它只会丢信息（gist 的作用是「细节已不可考」，
        # 可它本来就没细节）。
        return key
    clauses = _split_clauses(key)
    if not clauses:
        return truncate(key, limit)
    head = clauses[0]
    if len(head) < 6 and len(clauses) > 1:
        # 第一句太短（「嗯，」），多半是引子，取下一句。
        head = clauses[1]
    return truncate(head, limit)


def _split_clauses(text: str) -> list[str]:
    """按标点切句。冒号后面跟数字的不算断点（时间、URL）。"""

    out: list[str] = []
    buf: list[str] = []
    for index, ch in enumerate(text):
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if ch == ":" and nxt.isdigit():
            buf.append(ch)
            continue
        if ch in "，,。；;：:、！？!?\n":
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out




_STOPWORDS = frozenset(
    {
        # 单字虚词
        "的",
        "了",
        "是",
        "在",
        "和",
        "也",
        "就",
        "都",
        "而",
        "及",
        "与",
        "着",
        "或",
        "这",
        "那",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "个",
        "些",
        "不",
        "没",
        "很",
        "更",
        "最",
        "会",
        "要",
        "能",
        "有",
        "被",
        "把",
        "给",
        "让",
        # 常见双字虚词
        "我们",
        "他们",
        "自己",
        "这个",
        "那个",
        "什么",
        "怎么",
        "因为",
        "所以",
        "但是",
        "不过",
        "然后",
        "后来",
        "已经",
        "还是",
        "可以",
        "这样",
        "而且",
        "另外",
        "于是",
        "如果",
        "不过",
    }
)

_FUNCTION_CHARS = frozenset(
    "的了是在和也就都而及与着或这那我你他她它们个些不没很更最会要能有被把给让"
    "个所之其而且但却呢吧啊吗嘛哦嗯"
    # 疑问词：搜「什么」「怎么样」没有意义，但它们会命中一大片记忆
    "什么么怎哪谁何呀啦哇哈嘿唔呃额哎唉咦为啥呗咧噻样"
)


def is_function_token(token: str) -> bool:
    """这个 token 是不是「全由虚词拼起来的组合」。

    二元组索引会把「在的」「的是」「时的」这种组合也当成词。
    它们几乎出现在每一句中文里——拿它们去检索，
    等于跟整本语料库匹配，命中什么都不奇怪。

    所以**查询侧**要把这类 token 剔掉：
    两个字符都是虚词 → 丢掉。
    「存在」保留（存是实词），「在的」丢掉（两个都是虚词）。
    """

    if len(token) < 2:
        return True
    return all(ch in _FUNCTION_CHARS for ch in token)


def query_terms(text: str) -> tuple[list[str], list[str]]:
    """把查询拆成「多字词」和「单字」两组，并滤掉虚词组合。

    多字词走 BM25 主通道，单字走兜底通道（见 ``store.search_substring``）。
    两组都不含虚词组合——否则「现在的」这种查询会命中一大堆无关记忆。
    """

    key = normalize(text)
    if not key:
        return [], []

    multi: list[str] = []
    uni: list[str] = []
    for run in _CJK_RUN_RE.findall(key):
        if len(run) == 1:
            uni.append(run)
        else:
            multi.extend(run[i : i + 2] for i in range(len(run) - 1))
    for word in _WORD_RE.findall(key):
        lowered = word.lower()
        if len(lowered) == 1:
            uni.append(lowered)
        else:
            multi.append(lowered)
    multi = [t for t in dict.fromkeys(multi) if not is_function_token(t)]
    uni = [t for t in dict.fromkeys(uni) if t not in _FUNCTION_CHARS]
    return multi, uni


def content_words(text: str, limit: int = 8) -> list[str]:
    """抽取内容词（用于关键词图谱与关联召回）。

    中文按**二元组**切，不按单字。

    ⚠ 早先这里用的是单字切分，结果中文一个词都抽不出来：
    ``_CJK_RE.findall`` 返回的是单个汉字，``len(w) < 2`` 把它们全过滤了。
    而中文里单字几乎没有信息量（「的」「是」「我」到处都是），
    切单字等于没切。二元组才是中文能用得上的最小单位。
    """

    out: list[str] = []
    seen: set[str] = set()
    for token in bigrams(text):
        if len(token) < 2 or token in seen or token in _STOPWORDS:
            continue
        # 二元组会切出「的食」「辣的」这种半虚词组合——首尾挂着虚词的丢掉。
        if token[0] in _FUNCTION_CHARS or token[-1] in _FUNCTION_CHARS:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def join_tokens(tokens: Iterable[str]) -> str:
    """把 token 序列拼回 FTS 查询串。"""

    return " ".join(t for t in tokens if t)


_BRACKETS = {
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
    "「": "」",
    "『": "』",
    "《": "》",
    # 中文弯引号也得算：中文正文里写「简介“AI炼丹师，炼到什么发什么喵”」
    # 时，引号里的逗号不是分句点。漏了它，体检会对着一条好好的记忆报警。
    "“": "”",
    "‘": "’",
}
_CLOSERS = set(_BRACKETS.values())
# 句号级别的分隔符：一句就是一件事。
_STRONG_SEP = "。！？!?"
# 逗号级别的分隔符：短语之间的停顿。
# ⚠ 不含「、」：顿号是**并列项**的分隔符（「科目C/C++、Java、Python」
# 是一件事），拿它当切分点会把一个清单拆成四段孤零零的词。
_SOFT_SEP = "，,；;"
# 硬切时可以用的分隔符（顿号也算，因为它至少不切断一个词）。
_HARD_SEP = "，,、；;"
# 冒号单独处理：后面跟数字就不是分隔符（「10:00」「https://」）。
_COLON = "：:"
# 归一化（NFKC）会把全角冒号变成半角，所以「10:00」「https://」这类
# 会被误当成分句点。判断规则：冒号后面跟数字就不是分隔符。

# 「一句话」的判定口径——体检和拆分必须用同一把尺子，
# 否则体检说「一条记忆塞了几件事」，拆分却拆不动。
CLAUSE_SPLIT_RE = re.compile(r"[，,；;。！？!?]")


def clause_count(text: str) -> int:
    """一句话里有几个分句（顿号不算并列项，括号引号里的标点也不算）。

    口径必须和 ``split_atomic`` 一致——它切分时跳过括号引号里的标点，
    这里如果按裸标点切，「简介"AI炼丹师，炼到什么发什么喵"」会被算成两句，
    体检就会对着一条好好的记忆报警。
    """

    key = re.sub(r"\s+", " ", str(text or "")).strip()
    if not key:
        return 0
    return len([piece for piece, _ in _split_outside_brackets(key) if piece.strip()])


def split_atomic(text: str, *, max_chars: int = 40, max_clauses: int = 2) -> list[str]:
    """把一段长话拆成「一件事一条」。

    为什么要自己拆：LLM 不可用、或者模型没给出可解析的结果时，
    兜底不能只是「原样保留」——一段 284 字的话塞进记忆库，
    注入时只能截断（断在半句里），矛盾检测也抓不住（缺三元组）。
    规则拆分不如模型懂语义，但至少保证「一条一件事、一条不太长」。

    两条约束同时满足才算合格：

    * 不超过 ``max_chars`` 个字；
    * 不超过 ``max_clauses`` 个分句。

    切分点会跳过括号和引号里面的内容——
    「简介"AI炼丹师，炼到什么发什么喵"」里的逗号不是分句点。
    """

    # 这里**不**用 normalize()：NFKC 会把「（」「，」折成半角，
    # 拆出来的记忆就带着变形了的标点。折叠空白就够了。
    key = re.sub(r"\s+", " ", str(text or "")).strip()
    if not key:
        return []

    pieces = _split_outside_brackets(key)
    facts: list[str] = []
    buf = ""
    buf_sep = ""
    for piece, sep in pieces:
        candidate = (buf + buf_sep + piece) if buf else piece
        if len(candidate) <= max_chars and _count_clauses(candidate) <= max_clauses:
            buf = candidate
            buf_sep = sep
            continue
        if buf:
            facts.append(buf)
            buf = ""
        if len(piece) <= max_chars:
            buf = piece
            buf_sep = sep
        else:
            # 单个分句本身就超长 → 只能硬切，切点尽量落在分隔符上
            for part in _hard_split(piece, max_chars):
                facts.append(part)
    if buf:
        facts.append(buf)
    return facts


def _count_clauses(text: str) -> int:
    """分句数。口径与体检完全一致（见 ``clause_count``）。"""

    return clause_count(text)


def _split_outside_brackets(text: str) -> list[str]:
    """按标点切，但括号/引号里面的标点不算，冒号后面是数字也不算。

    返回 ``[(片段, 片段后面的分隔符)]``——分隔符要跟着走，
    合并回去时才能还原原来的标点（把「；」还原成「，」会改掉语气）。
    """

    out: list[tuple[str, str]] = []
    buf: list[str] = []
    stack: list[str] = []
    for index, ch in enumerate(text):
        if ch in _BRACKETS:
            stack.append(_BRACKETS[ch])
        elif ch in _CLOSERS and stack and stack[-1] == ch:
            stack.pop()
        nxt = text[index + 1] if index + 1 < len(text) else ""
        is_colon = ch in _COLON and not nxt.isdigit()
        if not stack and (ch in _STRONG_SEP or ch in _SOFT_SEP or is_colon):
            piece = "".join(buf).strip()
            if piece:
                out.append((piece, ch))
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append((tail, ""))
    return out


def _hard_split(text: str, limit: int) -> list[str]:
    """超长的单个分句：尽量在靠后的位置切开。"""

    if len(text) <= limit:
        return [text]
    out: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[: limit + 1]
        # 只在标点处切，绝不从数字/时间中间劈开
        # （「考4小时9:00-13:00」从中间切会变成「考4小时9」和「00-13:00」）。
        cut = max(window.rfind(sep) for sep in _HARD_SEP)
        if cut < limit // 2:
            cut = limit
        else:
            cut += 1
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


def longest_common_substring(a: str, b: str) -> str:
    """两个字符串的最长公共子串（字符级）。

    用来判断「对方是不是明确提到了这件事」：
    「还记得炸鸡吗」和「许杏玖喜欢吃炸鸡」的公共子串是「炸鸡」（2 字），
    而「今天吃了吗」和「昨天吃了炸鸡」只有「吃了」——同样 2 字，
    但前者的语义指向明确得多，所以阈值定在 2 字而不是更长。
    """

    s1 = normalize(a)
    s2 = normalize(b)
    if not s1 or not s2:
        return ""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    previous = [0] * (len(s1) + 1)
    best = 0
    best_end = 0
    for i in range(1, len(s2) + 1):
        current = [0] * (len(s1) + 1)
        ch = s2[i - 1]
        for j in range(1, len(s1) + 1):
            if s1[j - 1] == ch:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
                    best_end = i
        previous = current
    return s2[best_end - best : best_end]

