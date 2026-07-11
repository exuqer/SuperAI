"""Hive lifecycle, bounded hot context and tiered restoration.

Hives are task workspaces. They only retain references to global knowledge and
archive artifacts; they never become subject-specific copies of the Cosmos.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import (
    AccessScope,
    ContextEntry,
    DomainEvent,
    EvictionDecision,
    HiveState,
    HiveView,
    TaskContract,
    new_id,
    utcnow,
)
from .database import SqliteDatabase, json_dumps, json_loads
from .observability import TraceRecorder
from .storage import AccessDenied, ObjectStore


class HiveTransitionError(ValueError):
    pass


class CapacityError(RuntimeError):
    pass


_TRANSITIONS = {
    HiveState.ACTIVE: {HiveState.IDLE, HiveState.FROZEN, HiveState.COMPLETED, HiveState.FAILED},
    HiveState.IDLE: {HiveState.ACTIVE, HiveState.FROZEN, HiveState.COMPLETED, HiveState.ARCHIVED, HiveState.FAILED},
    HiveState.FROZEN: {HiveState.ACTIVE, HiveState.ARCHIVED, HiveState.FAILED},
    HiveState.COMPLETED: {HiveState.ARCHIVED, HiveState.ACTIVE},
    HiveState.ARCHIVED: {HiveState.ACTIVE},
    HiveState.FAILED: {HiveState.ACTIVE, HiveState.ARCHIVED},
}

_STOP_WORDS = {
    "и", "в", "во", "на", "с", "со", "о", "об", "для", "по", "что", "это", "как", "а", "но", "или",
    "мне", "мы", "ты", "вы", "я", "он", "она", "они", "the", "a", "an", "to", "of", "and", "or", "is",
    "вернемся", "вернемся", "вернуть", "обсудим", "пожалуйста", "помоги",
}
_RETURN_WORDS = {"вернемся", "вернуться", "верни", "продолжим", "return", "resume", "back"}


def topic_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[\w-]{3,}", text.lower(), flags=re.UNICODE):
        token = _normalise_token(raw)
        if token and token not in _STOP_WORDS and token not in terms:
            terms.append(token)
    return terms[:24]


def _normalise_token(token: str) -> str:
    # It is intentionally only a reversible-ish lexical aid, not a semantic
    # truth layer. Cosmos retrieval can later replace it with richer indices.
    for suffix in ("иями", "ями", "ами", "ого", "ему", "ами", "иях", "ях", "ах", "ой", "ей", "ом", "ем", "ов", "ев", "ам", "ям", "ую", "юю", "ий", "ый", "ая", "яя", "ое", "ее", "ы", "и", "а", "я", "у", "ю", "е", "о"):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class HiveManager:
    def __init__(self, database: SqliteDatabase, store: ObjectStore, traces: TraceRecorder) -> None:
        self.database = database
        self.store = store
        self.traces = traces

    def select_or_create(self, contract: TaskContract, trace_id: str) -> tuple[HiveView, str, list[dict[str, Any]]]:
        """Return a workspace and an explainable continue/create/restore decision."""
        rows = self.database.all(
            "SELECT * FROM hives WHERE tenant_id = ? AND conversation_id = ? AND state != ? ORDER BY updated_at DESC",
            (contract.tenant_id, contract.conversation_id, HiveState.ARCHIVED.value),
        )
        query_terms = set(topic_terms(contract.goal))
        asks_return = bool(query_terms & _RETURN_WORDS) or any(word in contract.goal.lower() for word in _RETURN_WORDS)
        scored: list[tuple[float, Dict[str, Any]]] = []
        for row in rows:
            if row["project_id"] != contract.project_id:
                continue
            previous_terms = set(json_loads(row["topic_json"], []))
            overlap = len(query_terms & previous_terms) / max(1, len(query_terms | previous_terms))
            score = overlap + (0.15 if row["project_id"] == contract.project_id else 0.0)
            if row["state"] in (HiveState.ACTIVE.value, HiveState.IDLE.value):
                score += 0.10
            if asks_return and overlap:
                score += 0.35
            scored.append((round(score, 4), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        alternatives = [
            {"hive_id": row["hive_id"], "score": score, "state": row["state"], "topics": json_loads(row["topic_json"], [])}
            for score, row in scored[:5]
        ]
        if scored:
            score, row = scored[0]
            threshold = 0.35 if asks_return else 0.42
            if score >= threshold:
                hive = self.get(row["hive_id"], contract.tenant_id, contract.project_id)
                decision = "restore" if hive.state == HiveState.FROZEN else "continue"
                if hive.state == HiveState.FROZEN:
                    hive = self.restore(hive.hive_id, contract.tenant_id, trace_id)
                elif hive.state != HiveState.ACTIVE:
                    hive = self.transition(hive.hive_id, HiveState.ACTIVE, contract.tenant_id, trace_id)
                # A continued Hive retains history but its current contract is
                # a new task revision, never the first task that created it.
                hive = self.update_contract(hive.hive_id, contract, trace_id)
                self._record_decision(contract, trace_id, decision, hive.hive_id, alternatives)
                return hive, decision, alternatives
        hive = self.create(contract, trace_id)
        self._record_decision(contract, trace_id, "create", hive.hive_id, alternatives)
        return hive, "create", alternatives

    def create(self, contract: TaskContract, trace_id: str) -> HiveView:
        hive_id = new_id("hive")
        now = utcnow()
        terms = topic_terms(contract.goal)
        state_data = {
            "goals": [contract.goal],
            "unresolved_commitments": [],
            "selected_knowledge_refs": [],
            "plan_refs": [],
            "critic_reports": [],
            "budget_ledger": {"hot_bytes": 0, "evicted_bytes": 0},
            "parent_hive_ids": [],
        }
        with self.database.transaction() as connection:
            # One active workspace per conversation is a convenient local MVP
            # policy; independent old workspaces remain idle and recoverable.
            if contract.project_id is None:
                connection.execute(
                    "UPDATE hives SET state = ?, updated_at = ? WHERE tenant_id = ? AND conversation_id = ? AND project_id IS NULL AND state = ?",
                    (HiveState.IDLE.value, now.isoformat(), contract.tenant_id, contract.conversation_id, HiveState.ACTIVE.value),
                )
            else:
                connection.execute(
                    "UPDATE hives SET state = ?, updated_at = ? WHERE tenant_id = ? AND conversation_id = ? AND project_id = ? AND state = ?",
                    (HiveState.IDLE.value, now.isoformat(), contract.tenant_id, contract.conversation_id, contract.project_id, HiveState.ACTIVE.value),
                )
            cursor = connection.execute(
                "INSERT INTO hives(hive_id, tenant_id, project_id, conversation_id, state, topic_json, contract_json, state_json, snapshot_id, version, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                (
                    hive_id,
                    contract.tenant_id,
                    contract.project_id,
                    contract.conversation_id,
                    HiveState.ACTIVE.value,
                    json_dumps(terms),
                    json_dumps(contract),
                    json_dumps(state_data),
                    1,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                raise HiveTransitionError("hive state changed while snapshot was being committed")
        self._event(contract, trace_id, "HiveCreated", {"hive_id": hive_id, "topics": terms})
        self.add_entry(
            hive_id,
            contract.tenant_id,
            store_name="GoalStore",
            content_type="task_goal",
            content={"goal": contract.goal, "constraints": contract.constraints},
            relevance=1.0,
            protected=True,
            trace_id=trace_id,
            idempotency_key="hive:%s:goal" % hive_id,
        )
        return self.get(hive_id, contract.tenant_id, contract.project_id)

    def get(
        self,
        hive_id: str,
        tenant_id: str,
        project_id: Optional[str] = None,
        *,
        include_warm: bool = True,
        enforce_project: bool = False,
    ) -> HiveView:
        row = self.database.one("SELECT * FROM hives WHERE hive_id = ? AND tenant_id = ?", (hive_id, tenant_id))
        if row is None:
            raise KeyError("hive not found")
        if enforce_project and row["project_id"] != project_id:
            raise AccessDenied("hive belongs to another project")
        layers = ("hot", "warm") if include_warm else ("hot",)
        placeholders = ",".join("?" for _ in layers)
        entries = self.database.all(
            "SELECT * FROM hive_entries WHERE hive_id = ? AND layer IN (%s) ORDER BY created_at" % placeholders,
            (hive_id, *layers),
        )
        return HiveView(
            hive_id=row["hive_id"],
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            conversation_id=row["conversation_id"],
            state=row["state"],
            contract=TaskContract.model_validate(json_loads(row["contract_json"])),
            topics=json_loads(row["topic_json"], []),
            state_data=json_loads(row["state_json"], {}),
            entries=[self._row_to_entry(entry) for entry in entries],
            snapshot_id=row["snapshot_id"],
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def update_contract(self, hive_id: str, contract: TaskContract, trace_id: str) -> HiveView:
        hive = self.get(hive_id, contract.tenant_id, contract.project_id)
        if contract.revision <= hive.contract.revision:
            contract.revision = hive.contract.revision + 1
        now = utcnow()
        self.database.execute(
            "UPDATE hives SET contract_json = ?, topic_json = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
            (json_dumps(contract), json_dumps(topic_terms(contract.goal)), now.isoformat(), hive_id),
        )
        self._event(contract, trace_id, "TaskContractRevised", {"hive_id": hive_id, "revision": contract.revision})
        return self.get(hive_id, contract.tenant_id, contract.project_id)

    def add_entry(
        self,
        hive_id: str,
        tenant_id: str,
        *,
        store_name: str,
        content_type: str,
        content: Dict[str, Any],
        relevance: float = 0.5,
        protected: bool = False,
        source_ref: Optional[str] = None,
        reconstruction_cost: float = 0.5,
        expiry_policy: str = "until_hive_complete",
        trace_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ContextEntry:
        hive = self.get(hive_id, tenant_id)
        if hive.state not in (HiveState.ACTIVE, HiveState.IDLE):
            raise HiveTransitionError("cannot add context to %s hive" % hive.state.value)
        encoded = json_dumps(content).encode("utf-8")
        entry = ContextEntry(
            hive_id=hive_id,
            store_name=store_name,
            content=content,
            content_type=content_type,
            size=len(encoded),
            relevance=relevance,
            protected=protected,
            source_ref=source_ref,
            reconstruction_cost=reconstruction_cost,
            expiry_policy=expiry_policy,
        )
        decisions: list[EvictionDecision] = []
        with self.database.transaction() as connection:
            if idempotency_key:
                existing = connection.execute(
                    "SELECT e.* FROM hive_entry_writes w JOIN hive_entries e ON e.entry_id = w.entry_id "
                    "WHERE w.hive_id = ? AND w.idempotency_key = ?",
                    (hive_id, idempotency_key),
                ).fetchone()
                if existing:
                    return self._row_to_entry(dict(existing))
            connection.execute(
                "INSERT INTO hive_entries(entry_id, hive_id, store_name, layer, content_json, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.entry_id,
                    hive_id,
                    store_name,
                    entry.layer,
                    json_dumps(content),
                    json_dumps(self._entry_metadata(entry)),
                    entry.created_at.isoformat(),
                ),
            )
            if idempotency_key:
                connection.execute(
                    "INSERT INTO hive_entry_writes(hive_id, idempotency_key, entry_id, created_at) VALUES (?, ?, ?, ?)",
                    (hive_id, idempotency_key, entry.entry_id, entry.created_at.isoformat()),
                )
            decisions = self._evict_to_budget_in_transaction(
                connection, hive_id, hive.contract.budget.memory_bytes
            )
            self._refresh_ledger_in_transaction(connection, hive_id)
        if trace_id and decisions:
            self._event(
                hive.contract,
                trace_id,
                "ContextEvicted",
                {"hive_id": hive_id, "decisions": [item.model_dump(mode="json") for item in decisions]},
            )
        return entry

    def enforce_capacity(self, hive_id: str, tenant_id: str, *, trace_id: Optional[str] = None) -> list[EvictionDecision]:
        hive = self.get(hive_id, tenant_id, include_warm=False)
        with self.database.transaction() as connection:
            decisions = self._evict_to_budget_in_transaction(
                connection, hive_id, hive.contract.budget.memory_bytes
            )
            self._refresh_ledger_in_transaction(connection, hive_id)
        if trace_id and decisions:
            self._event(
                hive.contract,
                trace_id,
                "ContextEvicted",
                {"hive_id": hive_id, "decisions": [item.model_dump(mode="json") for item in decisions]},
            )
        return decisions

    def _evict_to_budget_in_transaction(self, connection: Any, hive_id: str, budget: int) -> list[EvictionDecision]:
        rows = connection.execute(
            "SELECT * FROM hive_entries WHERE hive_id = ? AND layer = 'hot' ORDER BY created_at", (hive_id,)
        ).fetchall()
        entries = [self._row_to_entry(dict(row)) for row in rows]
        total = sum(entry.size for entry in entries)
        decisions: list[EvictionDecision] = []
        if total <= budget:
            return decisions
        # Exact duplicate cache entries are cheapest to remove. Other entries
        # move to warm context with an explainable score rather than vanishing.
        seen: set[str] = set()
        candidates: list[tuple[float, ContextEntry, str]] = []
        for entry in entries:
            if entry.protected:
                continue
            key = json_dumps(entry.content)
            if entry.store_name == "WorkingContextStore" and key in seen:
                candidates.append((-1.0, entry, "duplicate_cache"))
            else:
                seen.add(key)
                age_seconds = max(0.0, (utcnow() - entry.created_at).total_seconds())
                # lower scores leave hot memory first
                score = entry.relevance + min(1.0, entry.reconstruction_cost) * 0.25 - min(0.5, age_seconds / 86_400.0) - min(0.5, entry.size / max(1, budget))
                candidates.append((score, entry, "hot_budget_exceeded"))
        candidates.sort(key=lambda item: item[0])
        for score, entry, reason in candidates:
            if total <= budget:
                break
            connection.execute("UPDATE hive_entries SET layer = ? WHERE entry_id = ?", ("warm", entry.entry_id))
            decision = EvictionDecision(
                hive_id=hive_id,
                entry_id=entry.entry_id,
                reason_code=reason,
                score_before=score,
                score_after=0.0,
            )
            connection.execute(
                "INSERT INTO evictions(eviction_id, hive_id, entry_id, reason_code, score_before, score_after, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.eviction_id,
                    hive_id,
                    entry.entry_id,
                    reason,
                    score,
                    0.0,
                    decision.created_at.isoformat(),
                ),
            )
            decisions.append(decision)
            total -= entry.size
        if total > budget:
            # The caller sees a hard failure rather than an undisclosed budget
            # breach; protected context must be made smaller explicitly.
            raise CapacityError("protected hot context exceeds memory budget")
        return decisions

    def freeze(self, hive_id: str, tenant_id: str, trace_id: str) -> HiveView:
        hive = self.get(hive_id, tenant_id)
        if hive.state not in (HiveState.ACTIVE, HiveState.IDLE):
            raise HiveTransitionError("only active or idle hives can freeze")
        state = hive.model_dump(mode="json")
        now = utcnow().isoformat()

        def freeze_aggregate(connection: Any, snapshot: Any) -> None:
            cursor = connection.execute(
                "UPDATE hives SET state = ?, snapshot_id = ?, version = version + 1, updated_at = ? WHERE hive_id = ? AND state IN (?, ?)",
                (
                    HiveState.FROZEN.value,
                    snapshot.snapshot_id,
                    now,
                    hive_id,
                    HiveState.ACTIVE.value,
                    HiveState.IDLE.value,
                ),
            )
            if cursor.rowcount != 1:
                raise HiveTransitionError("hive state changed while snapshot was being committed")

        snapshot = self.store.create_snapshot(
            aggregate_type="Hive",
            aggregate_id=hive_id,
            sequence=hive.version,
            state=state,
            tenant_id=tenant_id,
            access_scope=AccessScope(
                tenant_id=tenant_id,
                project_id=hive.project_id,
                visibility="project" if hive.project_id else "tenant",
            ),
            after_insert=freeze_aggregate,
        )
        self._event(hive.contract, trace_id, "HiveFrozen", {"hive_id": hive_id, "snapshot_id": snapshot.snapshot_id})
        return self.get(hive_id, tenant_id)

    def restore(self, hive_id: str, tenant_id: str, trace_id: str) -> HiveView:
        hive = self.get(hive_id, tenant_id)
        if hive.state not in (HiveState.FROZEN, HiveState.ARCHIVED, HiveState.COMPLETED):
            return self.transition(hive_id, HiveState.ACTIVE, tenant_id, trace_id)
        # A snapshot is checked before it becomes active. Existing warm entries
        # are already persisted and avoid a full archive read in the normal path.
        snapshot = self.store.latest_snapshot("Hive", hive_id, tenant_id, project_id=hive.project_id)
        if snapshot:
            restored = self.store.restore_snapshot(snapshot, tenant_id, project_id=hive.project_id)
            try:
                restored_hive = HiveView.model_validate(restored)
            except Exception as exc:
                raise HiveTransitionError("snapshot schema is invalid: %s" % exc) from exc
            if restored_hive.tenant_id != tenant_id or restored_hive.hive_id != hive_id:
                raise AccessDenied("snapshot tenant mismatch")
            # Do not revive a reference the requester can no longer access.
            for entry in restored_hive.entries:
                if entry.source_ref:
                    self.store.get_metadata(entry.source_ref, tenant_id, project_id=hive.project_id)
            with self.database.transaction() as connection:
                connection.execute("DELETE FROM hive_entry_writes WHERE hive_id = ?", (hive_id,))
                connection.execute("DELETE FROM hive_entries WHERE hive_id = ?", (hive_id,))
                for entry in restored_hive.entries:
                    connection.execute(
                        "INSERT INTO hive_entries(entry_id, hive_id, store_name, layer, content_json, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            entry.entry_id,
                            hive_id,
                            entry.store_name,
                            entry.layer,
                            json_dumps(entry.content),
                            json_dumps(self._entry_metadata(entry)),
                            entry.created_at.isoformat(),
                        ),
                    )
                connection.execute(
                    "UPDATE hives SET state = ?, topic_json = ?, contract_json = ?, state_json = ?, snapshot_id = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
                    (
                        HiveState.ACTIVE.value,
                        json_dumps(restored_hive.topics),
                        json_dumps(restored_hive.contract),
                        json_dumps(restored_hive.state_data),
                        snapshot.snapshot_id,
                        utcnow().isoformat(),
                        hive_id,
                    ),
                )
                self._refresh_ledger_in_transaction(connection, hive_id)
            self._event(hive.contract, trace_id, "HiveRestored", {"hive_id": hive_id, "snapshot_id": snapshot.snapshot_id})
            return self.get(hive_id, tenant_id)
        return self.transition(hive_id, HiveState.ACTIVE, tenant_id, trace_id, event_kind="HiveRestored")

    def transition(
        self,
        hive_id: str,
        target: HiveState,
        tenant_id: str,
        trace_id: str,
        *,
        event_kind: str = "HiveStateChanged",
    ) -> HiveView:
        hive = self.get(hive_id, tenant_id)
        if target == hive.state:
            return hive
        if target not in _TRANSITIONS[hive.state]:
            raise HiveTransitionError("invalid transition %s -> %s" % (hive.state.value, target.value))
        now = utcnow().isoformat()
        self.database.execute(
            "UPDATE hives SET state = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
            (target.value, now, hive_id),
        )
        self._event(hive.contract, trace_id, event_kind, {"hive_id": hive_id, "from": hive.state.value, "to": target.value})
        return self.get(hive_id, tenant_id)

    def merge(self, hive_ids: Sequence[str], contract: TaskContract, trace_id: str) -> HiveView:
        """MVP merge is a new parent with links, never destructive data mixing."""
        if len(hive_ids) < 2:
            raise ValueError("at least two hives are required")
        for source_hive_id in hive_ids:
            self.get(
                source_hive_id,
                contract.tenant_id,
                contract.project_id,
                enforce_project=True,
            )
        parent = self.create(contract, trace_id)
        state = parent.state_data
        state["parent_hive_ids"] = list(hive_ids)
        self.database.execute(
            "UPDATE hives SET state_json = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
            (json_dumps(state), utcnow().isoformat(), parent.hive_id),
        )
        self._event(contract, trace_id, "HivesMerged", {"parent_hive_id": parent.hive_id, "source_hive_ids": list(hive_ids)})
        return self.get(parent.hive_id, contract.tenant_id, contract.project_id)

    def evictions(self, hive_id: str, tenant_id: str) -> list[EvictionDecision]:
        self.get(hive_id, tenant_id)
        return [
            EvictionDecision(
                eviction_id=row["eviction_id"],
                hive_id=row["hive_id"],
                entry_id=row["entry_id"],
                reason_code=row["reason_code"],
                score_before=row["score_before"],
                score_after=row["score_after"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in self.database.all("SELECT * FROM evictions WHERE hive_id = ? ORDER BY created_at DESC", (hive_id,))
        ]

    def record_execution(
        self,
        hive_id: str,
        tenant_id: str,
        *,
        plan_ref: Optional[str] = None,
        knowledge_refs: Optional[Sequence[str]] = None,
        critic_reports: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Store only compact references and reports, never duplicate Cosmos."""
        hive = self.get(hive_id, tenant_id)
        state = hive.state_data
        if plan_ref and plan_ref not in state.setdefault("plan_refs", []):
            state["plan_refs"].append(plan_ref)
        for reference in knowledge_refs or []:
            if reference not in state.setdefault("selected_knowledge_refs", []):
                state["selected_knowledge_refs"].append(reference)
        if critic_reports is not None:
            state["critic_reports"] = list(critic_reports)
        self.database.execute(
            "UPDATE hives SET state_json = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
            (json_dumps(state), utcnow().isoformat(), hive_id),
        )

    def _refresh_ledger(self, hive_id: str) -> None:
        row = self.database.one(
            "SELECT COALESCE(SUM(CASE WHEN layer = 'hot' THEN json_extract(metadata_json, '$.size') ELSE 0 END), 0) AS hot, "
            "COALESCE(SUM(CASE WHEN layer = 'warm' THEN json_extract(metadata_json, '$.size') ELSE 0 END), 0) AS warm "
            "FROM hive_entries WHERE hive_id = ?",
            (hive_id,),
        )
        hive_row = self.database.one("SELECT state_json FROM hives WHERE hive_id = ?", (hive_id,))
        if not hive_row:
            return
        state = json_loads(hive_row["state_json"], {})
        state["budget_ledger"] = {"hot_bytes": int(row["hot"]), "warm_bytes": int(row["warm"]), "evicted_bytes": int(row["warm"])}
        self.database.execute(
            "UPDATE hives SET state_json = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
            (json_dumps(state), utcnow().isoformat(), hive_id),
        )

    def _refresh_ledger_in_transaction(self, connection: Any, hive_id: str) -> None:
        rows = connection.execute(
            "SELECT layer, metadata_json FROM hive_entries WHERE hive_id = ?", (hive_id,)
        ).fetchall()
        hot = 0
        warm = 0
        for row in rows:
            size = int(json_loads(row["metadata_json"], {}).get("size", 0))
            if row["layer"] == "hot":
                hot += size
            elif row["layer"] == "warm":
                warm += size
        hive_row = connection.execute("SELECT state_json FROM hives WHERE hive_id = ?", (hive_id,)).fetchone()
        if not hive_row:
            return
        state = json_loads(hive_row["state_json"], {})
        state["budget_ledger"] = {"hot_bytes": hot, "warm_bytes": warm, "evicted_bytes": warm}
        connection.execute(
            "UPDATE hives SET state_json = ?, version = version + 1, updated_at = ? WHERE hive_id = ?",
            (json_dumps(state), utcnow().isoformat(), hive_id),
        )

    @staticmethod
    def _entry_metadata(entry: ContextEntry) -> Dict[str, Any]:
        return {
            "content_type": entry.content_type,
            "size": entry.size,
            "source_ref": entry.source_ref,
            "relevance": entry.relevance,
            "protected": entry.protected,
            "reconstruction_cost": entry.reconstruction_cost,
            "expiry_policy": entry.expiry_policy,
        }

    @staticmethod
    def _row_to_entry(row: Dict[str, Any]) -> ContextEntry:
        metadata = json_loads(row["metadata_json"], {})
        return ContextEntry(
            entry_id=row["entry_id"],
            hive_id=row["hive_id"],
            store_name=row["store_name"],
            layer=row["layer"],
            content=json_loads(row["content_json"], {}),
            content_type=metadata.get("content_type", "unknown"),
            size=metadata.get("size", 0),
            source_ref=metadata.get("source_ref"),
            relevance=metadata.get("relevance", 0.5),
            protected=metadata.get("protected", False),
            reconstruction_cost=metadata.get("reconstruction_cost", 0.5),
            expiry_policy=metadata.get("expiry_policy", "until_hive_complete"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _record_decision(
        self,
        contract: TaskContract,
        trace_id: str,
        decision: str,
        hive_id: str,
        alternatives: list[dict[str, Any]],
    ) -> None:
        self._event(
            contract,
            trace_id,
            "HiveSelectionDecided",
            {"decision": decision, "hive_id": hive_id, "candidates": alternatives},
        )

    def _event(self, contract: TaskContract, trace_id: str, kind: str, payload: Dict[str, Any]) -> None:
        self.traces.record_event(
            DomainEvent(
                id=new_id("env"),
                task_id=contract.task_id,
                trace_id=trace_id,
                tenant_id=contract.tenant_id,
                kind=kind,
                producer="HiveManager",
                payload=payload,
                correlation_id=contract.task_id,
            )
        )
