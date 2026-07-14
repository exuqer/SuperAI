"""Persistent chat sessions and the server-side hive/swarm orchestration."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from server.database import get_connection, init_db
from server.repositories.cloud_repository import CloudRepository, LayerRepository
from server.services.bee_algorithm import BeeAlgorithmConfig, BeeSwarm, FieldSampler, ForagingGoal, NectarPayload, _cosine
from server.tokenizer import tokenize_hierarchical


Broadcast = Callable[[str, Dict[str, Any]], Awaitable[None]]


def _now() -> str:
    return datetime.utcnow().isoformat()


class ChatSessionService:
    def __init__(self, config: Optional[BeeAlgorithmConfig] = None):
        self.config = config or BeeAlgorithmConfig()
        self.listeners: Dict[str, Set[Broadcast]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_sessions: Dict[str, str] = {}

    def _ensure_tables(self) -> None:
        with get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, turn_index INTEGER NOT NULL DEFAULT 0, max_cells INTEGER NOT NULL DEFAULT 24);
                CREATE TABLE IF NOT EXISTS chat_messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, turn_index INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, turn_index);
                CREATE TABLE IF NOT EXISTS swarm_turns (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, message_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', iteration INTEGER NOT NULL DEFAULT 0, goal_json TEXT NOT NULL DEFAULT '{}', metrics_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS swarm_events (id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_type TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, UNIQUE(turn_id, sequence));
                CREATE TABLE IF NOT EXISTS hive_cells (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, label TEXT NOT NULL, composition_json TEXT NOT NULL DEFAULT '{}', x REAL NOT NULL DEFAULT 0.0, y REAL NOT NULL DEFAULT 0.0, gravity REAL NOT NULL DEFAULT 0.0, visits INTEGER NOT NULL DEFAULT 0, source_id TEXT, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_hive_cells_session ON hive_cells(session_id, gravity DESC);
                """
            )
            conn.commit()

    def create_session(self) -> Dict[str, Any]:
        self._ensure_tables()
        session_id = str(uuid.uuid4())
        now = _now()
        with get_connection() as conn:
            conn.execute("INSERT INTO chat_sessions(id, created_at, updated_at, turn_index, max_cells) VALUES (?, ?, ?, 0, 24)", (session_id, now, now))
            conn.execute("INSERT INTO chat_messages(id, session_id, role, text, turn_index, created_at) VALUES (?, ?, 'assistant', ?, 0, ?)", (f"message-{uuid.uuid4().hex[:12]}", session_id, "Привет. Я собираю только релевантные локальные срезы глобального поля. Задайте вопрос, и рой начнёт поиск.", now))
            conn.commit()
        return self.get_state(session_id)

    def _session_exists(self, session_id: str) -> bool:
        with get_connection() as conn:
            return conn.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,)).fetchone() is not None

    def get_state(self, session_id: str) -> Dict[str, Any]:
        self._ensure_tables()
        with get_connection() as conn:
            session = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if not session:
                raise KeyError(session_id)
            messages = conn.execute("SELECT id, role, text, turn_index, created_at FROM chat_messages WHERE session_id = ? ORDER BY turn_index, created_at", (session_id,)).fetchall()
            cells = conn.execute("SELECT * FROM hive_cells WHERE session_id = ? ORDER BY gravity DESC", (session_id,)).fetchall()
            turn = conn.execute("SELECT * FROM swarm_turns WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
            events = conn.execute(
                "SELECT event_type, payload_json, created_at FROM swarm_events WHERE turn_id = ? ORDER BY sequence DESC LIMIT 12",
                (turn["id"],),
            ).fetchall() if turn else []
        swarm = self._swarm_snapshot(turn, events)
        if not swarm.get("context_areas"):
            swarm["context_areas"] = FieldSampler({"word_form", "lexeme", "concept", "scene"}).context_areas()
        return {
            "session": {"id": session["id"], "turn_index": session["turn_index"], "max_cells": session["max_cells"], "updated_at": session["updated_at"]},
            "messages": [dict(row) for row in messages],
            "hive": [self._cell_dict(row) for row in cells],
            "turn": dict(turn) if turn else None,
            "swarm": swarm,
        }

    @staticmethod
    def _swarm_snapshot(turn: Any, events: List[Any]) -> Dict[str, Any]:
        if not turn:
            return {}
        latest_iteration: Dict[str, Any] = {}
        event_log: List[Dict[str, Any]] = []
        for row in events:
            payload = json.loads(row["payload_json"] or "{}")
            event_log.append({"type": row["event_type"], "payload": payload, "created_at": row["created_at"]})
            if row["event_type"] == "swarm_iteration_completed" and not latest_iteration:
                latest_iteration = payload
        return {
            "status": turn["status"],
            "goal": json.loads(turn["goal_json"] or "{}"),
            "iteration": latest_iteration.get("iteration", turn["iteration"]),
            "sources": latest_iteration.get("sources", []),
            "bees": latest_iteration.get("bees", []),
            "metrics": latest_iteration.get("metrics", json.loads(turn["metrics_json"] or "{}")),
            "events": list(reversed(event_log)),
        }

    @staticmethod
    def _cell_dict(row: Any) -> Dict[str, Any]:
        return {"id": row["id"], "label": row["label"], "composition": json.loads(row["composition_json"] or "{}"), "x": row["x"], "y": row["y"], "gravity": row["gravity"], "strength": row["gravity"], "visits": row["visits"], "source_id": row["source_id"]}

    def subscribe(self, session_id: str, callback: Broadcast) -> None:
        self.listeners.setdefault(session_id, set()).add(callback)

    def unsubscribe(self, session_id: str, callback: Broadcast) -> None:
        self.listeners.get(session_id, set()).discard(callback)

    async def _broadcast(self, session_id: str, event: Dict[str, Any]) -> None:
        dead: List[Broadcast] = []
        for listener in list(self.listeners.get(session_id, set())):
            try:
                await listener(session_id, event)
            except Exception:
                dead.append(listener)
        for listener in dead:
            self.unsubscribe(session_id, listener)

    def start_message(self, session_id: str, text: str) -> Dict[str, Any]:
        self._ensure_tables()
        text = text.strip()
        if not text or not self._session_exists(session_id):
            raise ValueError("session or message is invalid")
        with get_connection() as conn:
            session = conn.execute("SELECT turn_index FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            active = conn.execute("SELECT 1 FROM swarm_turns WHERE session_id = ? AND status IN ('queued', 'running') LIMIT 1", (session_id,)).fetchone()
            if active:
                raise ValueError("a swarm turn is already running")
            turn_index = int(session["turn_index"]) + 1
            message_id = f"message-{uuid.uuid4().hex[:12]}"
            turn_id = f"turn-{uuid.uuid4().hex[:12]}"
            now = _now()
            conn.execute("UPDATE chat_sessions SET turn_index = ?, updated_at = ? WHERE id = ?", (turn_index, now, session_id))
            conn.execute("INSERT INTO chat_messages(id, session_id, role, text, turn_index, created_at) VALUES (?, ?, 'user', ?, ?, ?)", (message_id, session_id, text, turn_index, now))
            conn.execute("INSERT INTO swarm_turns(id, session_id, message_id, status, created_at, updated_at) VALUES (?, ?, ?, 'queued', ?, ?)", (turn_id, session_id, message_id, now, now))
            conn.commit()
        task = asyncio.create_task(self._run_turn(session_id, turn_id, message_id, text, turn_index))
        self.tasks[turn_id] = task
        self.task_sessions[turn_id] = session_id
        task.add_done_callback(lambda _: (self.tasks.pop(turn_id, None), self.task_sessions.pop(turn_id, None)))
        return {"turn_id": turn_id, "message_id": message_id, "turn_index": turn_index}

    def _build_goal(self, session_id: str, message_id: str, text: str, turn_index: int) -> ForagingGoal:
        tokenization = tokenize_hierarchical(text)
        words = [token.normalized.casefold() for token in tokenization.all_tokens]
        cloud_repo, layer_repo = CloudRepository(), LayerRepository()
        query: Dict[str, float] = {}
        cloud_ids: List[int] = []
        for layer_name in ("word_form", "lexeme", "concept"):
            layer = layer_repo.get_by_name(layer_name)
            if not layer:
                continue
            for word in words:
                cloud = cloud_repo.get_by_canonical_name(layer.id, word)
                if cloud:
                    query[word] = query.get(word, 0.0) + (1.0 if layer_name != "concept" else 0.7)
                    if cloud.id not in cloud_ids:
                        cloud_ids.append(cloud.id)
        total = sum(query.values())
        if total:
            query = {key: value / total for key, value in query.items()}
        scene_ids: List[int] = []
        scene_layer = layer_repo.get_by_name("scene")
        if scene_layer:
            scene = cloud_repo.get_by_canonical_name(scene_layer.id, " ".join(words))
            if scene:
                scene_ids.append(scene.id)
        with get_connection() as conn:
            cells = conn.execute("SELECT label, composition_json, gravity FROM hive_cells WHERE session_id = ? ORDER BY gravity DESC", (session_id,)).fetchall()
            previous = conn.execute("SELECT id, turn_index FROM chat_messages WHERE session_id = ? AND role = 'user' ORDER BY turn_index DESC LIMIT 8", (session_id,)).fetchall()
        dialogue: Dict[str, float] = {}
        for cell in cells:
            weight = max(0.0, float(cell["gravity"]))
            for key, value in json.loads(cell["composition_json"] or "{}").items():
                dialogue[key] = dialogue.get(key, 0.0) + weight * float(value)
            dialogue[cell["label"].casefold()] = dialogue.get(cell["label"].casefold(), 0.0) + weight
        dtotal = sum(dialogue.values())
        if dtotal:
            dialogue = {key: value / dtotal for key, value in dialogue.items()}
        return ForagingGoal(message_id, cloud_ids, scene_ids, query, dialogue, previous_message_weights={str(row["id"]): 0.82 ** (turn_index - int(row["turn_index"])) for row in previous})

    async def _run_turn(self, session_id: str, turn_id: str, message_id: str, text: str, turn_index: int) -> None:
        goal = self._build_goal(session_id, message_id, text, turn_index)
        gravity_cells = self._apply_message_gravity(session_id, goal)
        sequence = 0
        with get_connection() as conn:
            conn.execute("UPDATE swarm_turns SET status = 'running', goal_json = ?, updated_at = ? WHERE id = ?", (json.dumps(goal.to_dict(), ensure_ascii=False), _now(), turn_id))
            conn.commit()

        async def emit(event_type: str, payload: Dict[str, Any]) -> None:
            nonlocal sequence
            sequence += 1
            if event_type == "nectar_collected":
                cell = self.deposit_payload(session_id, payload["payload"])
                payload = {**payload, "cell": cell}
            if event_type == "swarm_iteration_completed":
                with get_connection() as conn:
                    conn.execute("UPDATE swarm_turns SET iteration = ?, metrics_json = ?, updated_at = ? WHERE id = ?", (payload.get("iteration", 0), json.dumps(payload.get("metrics", {})), _now(), turn_id))
                    conn.commit()
            event = {"turn_id": turn_id, "sequence": sequence, "type": event_type, "payload": payload}
            with get_connection() as conn:
                conn.execute("INSERT OR REPLACE INTO swarm_events(turn_id, sequence, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)", (turn_id, sequence, event_type, json.dumps(payload, ensure_ascii=False), _now()))
                conn.commit()
            await self._broadcast(session_id, event)

        try:
            if gravity_cells:
                await emit("hive_gravity_updated", {"cells": gravity_cells, "reason": "current_message"})
            swarm = BeeSwarm(goal, session_id, turn_index, self.config)
            await swarm.run(emit)
            summary = self._make_summary(session_id, text, bool(goal.query_composition or goal.dialogue_composition))
            with get_connection() as conn:
                conn.execute("UPDATE swarm_turns SET status = 'completed', updated_at = ? WHERE id = ?", (_now(), turn_id))
                conn.execute("INSERT INTO chat_messages(id, session_id, role, text, turn_index, created_at) VALUES (?, ?, 'assistant', ?, ?, ?)", (f"message-{uuid.uuid4().hex[:12]}", session_id, summary, turn_index, _now()))
                conn.commit()
            await emit("assistant_message", {"text": summary})
        except Exception as exc:
            with get_connection() as conn:
                conn.execute("UPDATE swarm_turns SET status = 'failed', metrics_json = ?, updated_at = ? WHERE id = ?", (json.dumps({"error": str(exc)}), _now(), turn_id))
                conn.commit()
            await emit("swarm_failed", {"error": str(exc)})

    def _apply_message_gravity(self, session_id: str, goal: ForagingGoal) -> List[Dict[str, Any]]:
        """Let the current message reinforce matching cells without creating nectar."""
        changed: List[Dict[str, Any]] = []
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM hive_cells WHERE session_id = ?", (session_id,)).fetchall()
            for row in rows:
                composition = json.loads(row["composition_json"] or "{}")
                alignment = _cosine(goal.query_composition, composition)
                if alignment <= 0.05:
                    gravity = row["gravity"] * 0.86
                else:
                    gravity = min(1.0, row["gravity"] * 0.86 + alignment * 0.16)
                if gravity < 0.08:
                    conn.execute("DELETE FROM hive_cells WHERE id = ?", (row["id"],))
                    continue
                conn.execute("UPDATE hive_cells SET gravity = ?, updated_at = ? WHERE id = ?", (gravity, _now(), row["id"]))
                changed.append({"id": row["id"], "label": row["label"], "composition": composition, "x": row["x"], "y": row["y"], "gravity": gravity, "strength": gravity, "visits": row["visits"], "source_id": row["source_id"]})
            conn.commit()
        return changed

    def deposit_payload(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        composition = payload.get("composition", {})
        if not composition:
            return {}
        label = " · ".join(key for key, _ in sorted(composition.items(), key=lambda item: item[1], reverse=True)[:3])
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM hive_cells WHERE session_id = ? ORDER BY gravity DESC", (session_id,)).fetchall()
            for row in rows:
                existing = json.loads(row["composition_json"] or "{}")
                if _cosine(existing, composition) >= 0.82:
                    merged = {key: existing.get(key, 0.0) * 0.65 + composition.get(key, 0.0) * 0.35 for key in set(existing) | set(composition)}
                    gravity = min(1.0, row["gravity"] * 0.92 + float(payload.get("strength", 0.0)) * 0.42)
                    conn.execute("UPDATE hive_cells SET label = ?, composition_json = ?, gravity = ?, visits = visits + 1, source_id = ?, updated_at = ? WHERE id = ?", (label, json.dumps(merged, ensure_ascii=False), gravity, payload.get("source_id"), _now(), row["id"]))
                    conn.commit()
                    return self._cell_by_id(row["id"])
            index = len(rows)
            if index >= 24:
                weakest = min(rows, key=lambda row: row["gravity"])
                conn.execute("DELETE FROM hive_cells WHERE id = ?", (weakest["id"],))
                index -= 1
            for row in rows:
                if row["gravity"] < 0.08:
                    conn.execute("DELETE FROM hive_cells WHERE id = ?", (row["id"],))
            angle = (index * 2.399963) % (math.pi * 2)
            radius = 18.0 + 28.0 * math.sqrt(max(1, index))
            cell_id = f"cell-{uuid.uuid4().hex[:12]}"
            conn.execute("INSERT INTO hive_cells(id, session_id, label, composition_json, x, y, gravity, visits, source_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)", (cell_id, session_id, label, json.dumps(composition, ensure_ascii=False), 50 + math.cos(angle) * radius, 50 + math.sin(angle) * radius, min(1.0, float(payload.get("strength", 0.0)) * 0.55), payload.get("source_id"), _now()))
            conn.commit()
        return self._cell_by_id(cell_id)

    def _cell_by_id(self, cell_id: str) -> Dict[str, Any]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM hive_cells WHERE id = ?", (cell_id,)).fetchone()
        return self._cell_dict(row) if row else {}

    def _make_summary(self, session_id: str, text: str, has_goal: bool) -> str:
        with get_connection() as conn:
            rows = conn.execute("SELECT label, gravity FROM hive_cells WHERE session_id = ? ORDER BY gravity DESC LIMIT 6", (session_id,)).fetchall()
        if not has_goal and not rows:
            return "В глобальном поле не найдено известных понятий для этого сообщения. Обучение поля выполняется отдельно, поэтому улей не изменён."
        if not rows:
            return "Рой не нашёл источник выше порога качества. Рабочая память осталась без нового нектара."
        context = ", ".join(f"{row['label']} ({round(row['gravity'] * 100)}%)" for row in rows)
        return f"Собран локальный контекст по теме «{text[:64]}{'…' if len(text) > 64 else ''}». В рабочем улье: {context}. Слабые и повторные источники вытеснены."

    def reset(self, session_id: str) -> None:
        for turn_id, task in list(self.tasks.items()):
            if self.task_sessions.get(turn_id) == session_id:
                task.cancel()
        with get_connection() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()
        self.listeners.pop(session_id, None)


chat_service = ChatSessionService()
