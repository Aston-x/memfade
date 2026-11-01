"""离线整理（睡眠）。

每隔一段时间做一轮，把放着不管就会变脏的地方收拾干净：

1. **衰减重算**——按锚点重算有效强度，解出新的淡化档位；
2. **压缩成 gist**——淡出到 gisted 档的记忆，把原文压成一行摘要；
3. **合并重复**——近似重复的记忆合成一条，被合并方的原文留档；
4. **矛盾仲裁**——同主语同谓语但宾语不同的，新事实取代旧事实，
   旧事实标记为 superseded（保留原文做证据链，不删）；
5. **物理清理**——归档超过指定天数的才真正删（默认永不删）。

## 幂等

第 1 步永远按「距上次被想起多少天」重算，而不是每天乘一次系数，
所以这一轮跑两次、跑十次、隔一天再跑，结果完全一样。
这比 UPDATE ... SET strength = strength * 0.99 健壮得多：
那种写法在重启、多实例、定时器抖动时会重复衰减，把记忆洗白。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional

from .config import MemFadeConfig
from .decay import DEFAULT_POLICY, FadePolicy, effective_strength, resolve_status
from .dedup import choose_survivor, find_duplicates, merge_memories
from .extract import MemoryExtractor
from .store import MemoryStore
from .textutil import fts_text
from .types import Memory, MemoryStatus


@dataclass
class ConsolidationReport:
    """一轮整理的结果。"""

    scanned: int = 0
    decayed: int = 0
    status_changed: int = 0
    gisted: int = 0
    merged: int = 0
    superseded: int = 0
    purged: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "decayed": self.decayed,
            "status_changed": self.status_changed,
            "gisted": self.gisted,
            "merged": self.merged,
            "superseded": self.superseded,
            "purged": self.purged,
            "errors": list(self.errors),
        }


class ConsolidationService:
    """整理服务。"""

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemFadeConfig] = None,
        policy: Optional[FadePolicy] = None,
        extractor: Optional[MemoryExtractor] = None,
    ) -> None:
        self.store = store
        self.config = config or MemFadeConfig()
        self.policy = policy or DEFAULT_POLICY
        self.extractor = extractor

    async def run(self, scope_key: Optional[str] = None, *, now: Optional[float] = None) -> ConsolidationReport:
        """跑一轮整理。scope_key 为 None 时整理全库。"""

        ts = float(now if now is not None else time.time())
        report = ConsolidationReport()
        limit = self.config.maintenance.max_rows_per_pass

        memories = self._load(scope_key, limit)
        report.scanned = len(memories)

        decayed, changed, gisted = await self._decay_pass(memories, ts, report)
        report.decayed = decayed
        report.status_changed = changed
        report.gisted = gisted
        if self.config.maintenance.enable_merge:
            report.merged = self._merge_pass(scope_key, report)
        if self.config.maintenance.enable_supersede:
            report.superseded = self._supersede_pass(scope_key)
        report.purged += self._trim_pass(scope_key)
        report.purged += self._purge_pass(ts)
        return report

    def _load(self, scope_key: Optional[str], limit: int) -> list[Memory]:
        if scope_key:
            return self.store.iter_scope(scope_key, statuses=("active", "fading", "gisted"), limit=limit)
        return self.store.iter_maintenance(limit=limit)

    async def _decay_pass(
        self,
        memories: list[Memory],
        now: float,
        report: ConsolidationReport,
    ) -> tuple:
        """重算有效强度与淡化档位。"""

        decayed = 0
        changed = 0
        gisted = 0
        for mem in memories:
            if mem.pinned:
                continue
            eff = effective_strength(mem, now, self.policy)
            status = resolve_status(mem, now, self.policy)
            # 归档是一条单向门：一旦 archive 就不再自动复活（除非被召回）
            if mem.status is MemoryStatus.ARCHIVED and status is not MemoryStatus.ARCHIVED:
                status = MemoryStatus.ARCHIVED
            if status is mem.status and abs(eff - mem.strength) < 1e-9:
                continue
            decayed += 1
            # ⚠ 必须**复制**再改：`updated = mem` 是同一个对象，
            # 改完 status 再拿 mem.status 比较，结果永远相等——
            # 「有多少条换了档位」这个计数会恒为 0，看不出任何变化。
            updated = replace(mem, status=status, updated_at=now)
            summary = mem.summary
            if status is MemoryStatus.GISTED and not summary:
                summary = await self._make_gist(mem)
                if summary:
                    gisted += 1
            updated.summary = summary
            try:
                self.store.update(updated, fts_text(updated.content))
            except Exception as exc:
                report.errors.append(f"更新 #{mem.id} 失败: {exc}")
                continue
            if status is not mem.status:
                changed += 1
        return decayed, changed, gisted

    async def _make_gist(self, mem: Memory) -> str:
        """生成 gist 摘要。"""

        if not self.config.fade.gist_enabled:
            return ""
        if self.extractor is not None:
            try:
                return await self.extractor.summarize(
                    mem.content, max_chars=self.config.fade.gist_max_chars
                )
            except Exception:
                pass
        from .textutil import first_clause

        return first_clause(mem.content, self.config.fade.gist_max_chars)

    def _merge_pass(self, scope_key: Optional[str], report: ConsolidationReport) -> int:
        """合并近似重复。"""

        merged = 0
        if scope_key:
            candidates = self.store.iter_scope(
                scope_key, statuses=("active", "fading", "gisted"), limit=2000
            )
        else:
            candidates = self.store.iter_maintenance(limit=2000)
        for a, b in find_duplicates(candidates, threshold=self.config.retrieval.merge_threshold):
            survivor = choose_survivor(a, b)
            loser = b if survivor is a else a
            combined = merge_memories(survivor, loser)
            try:
                self.store.update(combined, fts_text(combined.content))
                self.store.delete(loser.id)
                merged += 1
            except Exception as exc:
                report.errors.append(f"合并 #{a.id}/#{b.id} 失败: {exc}")
        return merged

    def _supersede_pass(self, scope_key: Optional[str]) -> int:
        """矛盾仲裁：新事实取代旧事实。"""

        if not scope_key:
            return 0
        count = 0
        for older, newer in self.store.find_conflicts(scope_key):
            if older.id == newer.id:
                continue
            try:
                self.store.mark_superseded(older.id, newer.id)
                count += 1
            except Exception:
                continue
        return count

    def _purge_pass(self, now: float) -> int:
        """物理清理归档很久的记忆。默认不删。"""

        days = int(self.config.fade.archive_after_days)
        if days <= 0:
            return 0
        cutoff = now - days * 86400.0
        purged = 0
        for mem in self.store.iter_archived(cutoff, limit=500):
            if self.store.delete(mem.id):
                purged += 1
        return purged

    def _trim_pass(self, scope_key: Optional[str]) -> int:
        """超出上限时淘汰最弱的记忆。

        淘汰顺序：已归档 → 残影 → 淡出中 → 活跃。
        置顶的、以及被反复提起过的，不动。
        """

        cap = int(self.config.maintenance.max_memories_per_scope)
        if cap <= 0 or not scope_key:
            return 0
        memories = self.store.iter_scope(scope_key, limit=cap * 4 + 1000)
        if len(memories) <= cap:
            return 0
        status_rank = {
            MemoryStatus.ARCHIVED.value: 0,
            MemoryStatus.GISTED.value: 1,
            MemoryStatus.FADING.value: 2,
            MemoryStatus.ACTIVE.value: 3,
            MemoryStatus.SUPERSEDED.value: 0,
        }
        removable = [m for m in memories if not m.pinned]
        removable.sort(
            key=lambda m: (
                status_rank.get(m.status.value, 3),
                m.strength,
                m.importance,
                m.recall_count,
                m.updated_at,
            )
        )
        excess = len(memories) - cap
        removed = 0
        for mem in removable:
            if removed >= excess:
                break
            if self.store.delete(mem.id):
                removed += 1
        return removed

