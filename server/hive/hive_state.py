from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from server.memory import ThermoGravityMemory
from server.spaces import ConceptSpace, EventSpace, MorphemeSpace, SymbolSpace, WordSpace
from server.v2.repository import V2Repository, decode, encode, utcnow


class SpaceRegistry:
    def __init__(self) -> None:
        self.event = EventSpace()
        self.concept = ConceptSpace()
        self.word = WordSpace()
        self.morpheme = MorphemeSpace()
        self.symbol = SymbolSpace()

    @property
    def all(self) -> dict[str, Any]:
        return {
            self.event.level.value: self.event,
            self.concept.level.value: self.concept,
            self.word.level.value: self.word,
            self.morpheme.level.value: self.morpheme,
            self.symbol.level.value: self.symbol,
        }

    def get(self, name: str) -> Any:
        return self.all[name]

    def to_dict(self) -> dict[str, Any]:
        return {name: space.to_dict() for name, space in self.all.items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpaceRegistry":
        registry = cls()
        for name, payload in value.items():
            if name in registry.all:
                registry.get(name).load(payload)
        return registry


@dataclass
class HiveState:
    hive_id: str
    turn: int = 0
    spaces: SpaceRegistry = field(default_factory=SpaceRegistry)
    memory: ThermoGravityMemory = field(default_factory=ThermoGravityMemory)
    active_tasks: list[dict[str, Any]] = field(default_factory=list)
    nectar_packets: list[dict[str, Any]] = field(default_factory=list)
    vertical_transitions: list[dict[str, Any]] = field(default_factory=list)
    factories: list[dict[str, Any]] = field(default_factory=list)
    reasoning_ticks: list[dict[str, Any]] = field(default_factory=list)
    current_trace: dict[str, Any] = field(default_factory=dict)
    answer: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, int] = field(
        default_factory=lambda: {
            "max_bees_per_slot": 12,
            "max_downward_depth": 3,
            "max_reasoning_ticks": 12,
            "max_candidates_per_level": 8,
            "max_external_requests": 1,
            "max_cold_clusters": 12,
            "max_event_objects": 128,
            "max_concept_objects": 192,
            "max_word_objects": 256,
            "max_morpheme_objects": 384,
            "max_symbol_objects": 768,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "hive_id": self.hive_id,
            "turn": self.turn,
            "spaces": self.spaces.to_dict(),
            "memory": self.memory.to_dict(),
            "active_tasks": self.active_tasks,
            "nectar_packets": self.nectar_packets,
            "vertical_transitions": self.vertical_transitions,
            "factories": self.factories,
            "reasoning_ticks": self.reasoning_ticks,
            "current_trace": self.current_trace,
            "answer": self.answer,
            "limits": self.limits,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HiveState":
        state = cls(hive_id=str(value["hive_id"]))
        state.turn = int(value.get("turn", 0))
        state.spaces = SpaceRegistry.from_dict(value.get("spaces", {}))
        state.memory = ThermoGravityMemory.from_dict(value.get("memory", {}))
        state.active_tasks = list(value.get("active_tasks", []))
        state.nectar_packets = list(value.get("nectar_packets", []))
        state.vertical_transitions = list(value.get("vertical_transitions", []))
        state.factories = list(value.get("factories", []))
        state.reasoning_ticks = list(value.get("reasoning_ticks", []))
        state.current_trace = dict(value.get("current_trace", {}))
        state.answer = dict(value.get("answer", {}))
        state.limits.update(value.get("limits", {}))
        return state


class HiveStateStore:
    def __init__(self, repository: V2Repository | None = None) -> None:
        self.repository = repository or V2Repository()

    @staticmethod
    def ensure_schema(conn: Any) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hive_multilevel_states (
                hive_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS hive_multilevel_traces (
                id INTEGER PRIMARY KEY,
                hive_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                trace_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(hive_id) REFERENCES hives(id) ON DELETE CASCADE,
                UNIQUE(hive_id, turn)
            );
            CREATE INDEX IF NOT EXISTS hive_multilevel_trace_idx
                ON hive_multilevel_traces(hive_id, turn DESC);
            """
        )

    def load(self, hive_id: str) -> HiveState:
        with self.repository.transaction() as conn:
            self.ensure_schema(conn)
            if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            row = conn.execute(
                "SELECT state_json FROM hive_multilevel_states WHERE hive_id=?", (hive_id,)
            ).fetchone()
        return HiveState.from_dict(decode(row["state_json"])) if row else HiveState(hive_id)

    def save(self, state: HiveState, *, save_trace: bool = True) -> None:
        now = utcnow()
        with self.repository.transaction() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """INSERT INTO hive_multilevel_states(hive_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(hive_id) DO UPDATE SET state_json=excluded.state_json,
                updated_at=excluded.updated_at""",
                (state.hive_id, encode(state.to_dict()), now),
            )
            if save_trace and state.current_trace:
                conn.execute(
                    """INSERT INTO hive_multilevel_traces(hive_id, turn, trace_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(hive_id, turn) DO UPDATE SET trace_json=excluded.trace_json""",
                    (state.hive_id, state.turn, encode(state.current_trace), now),
                )

    def traces(self, hive_id: str) -> list[dict[str, Any]]:
        with self.repository.transaction() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                "SELECT turn, trace_json, created_at FROM hive_multilevel_traces WHERE hive_id=? ORDER BY turn",
                (hive_id,),
            ).fetchall()
        return [
            {
                "turn": row["turn"],
                "trace": decode(row["trace_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
