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
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def normalize(text: str) -> str:
    """NFKC 规范化 + 折叠空白。保留标点，用于展示。"""

    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", folded).strip()


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
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]+", key):
        if len(run) == 1:
            out.append(run)
        else:
            for i in range(len(run) - 1):
                out.append(run[i : i + 2])

    out.extend(word.lower() for word in _WORD_RE.findall(key))
    return out


def fts_text(text: str) -> str:
    """喂给 FTS5 的索引串：二元组之间用空格隔开。"""

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

    key = normalize(text)
    if limit <= 0 or len(key) <= limit:
        return key
    return key[: max(0, limit - len(ellipsis))] + ellipsis


def first_clause(text: str, limit: int = 40) -> str:
    """取第一个子句并截断——LLM 不可用时做 gist 的兜底策略。"""

    if not text:
        return ""
    parts = re.split(r"[，,。；;：:、]", normalize(text))
    head = next((p.strip() for p in parts if p.strip()), normalize(text))
    return truncate(head, limit)


def content_words(text: str, limit: int = 8) -> list[str]:
    """抽取内容词（用于关键词图谱与关联召回）。"""

    words = [w.lower() for w in _WORD_RE.findall(normalize(text))]
    cjk = _CJK_RE.findall(normalize(text))
    out: list[str] = []
    seen: set[str] = set()
    for w in words + cjk:
        if len(w) < 2 or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def join_tokens(tokens: Iterable[str]) -> str:
    """把 token 序列拼回 FTS 查询串。"""

    return " ".join(t for t in tokens if t)

