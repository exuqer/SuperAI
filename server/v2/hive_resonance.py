"""Temporary, spatial resonance sessions for a hive.

Lexical matching is intentionally delegated to :mod:`server.v2.resonance`.
Nothing in this module mutates a cloud, placement, or other persistent memory.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from .repository import V2Repository, decode, encode, utcnow
from .resonance import LexicalCandidateResolver


@dataclass
class ResonanceConfig:
    max_ticks: int = 8
    decay: float = .82
    temperature: float = .25
    activation_threshold: float = .08
    candidate_threshold: float = .20
    max_seed_candidates: int = 12
    stability_threshold: float = .02
    stable_ticks_required: int = 2
    distance_sigma: float = .40
    velocity_damping: float = .70
    max_velocity: float = .05
    max_displacement: float = .20
    global_memory_weight: float = .70
    energy_budget: float = 1.0
    suppression_threshold: float = .15

    @classmethod
    def from_values(cls, values: Optional[Dict[str, Any]] = None) -> "ResonanceConfig":
        allowed = {key: value for key, value in (values or {}).items() if key in cls.__dataclass_fields__}
        return cls(**allowed)


class HiveResonanceEngine:
    """Runs bounded excitation propagation in an isolated session state."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        self.resolver = LexicalCandidateResolver(self.repository)

    def create(self, hive_id: str, input_text: str, *, temperature: Optional[float] = None,
               max_ticks: Optional[int] = None, use_global_memory: bool = True,
               save_snapshots: bool = True, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        values = dict(config or {})
        if temperature is not None:
            values["temperature"] = temperature
        if max_ticks is not None:
            values["max_ticks"] = max_ticks
        cfg = ResonanceConfig.from_values(values)
        candidates = self.resolver.resolve(hive_id, input_text, use_global=use_global_memory,
                                           max_candidates=cfg.max_seed_candidates)
        context = self._dialogue_context(hive_id, input_text)
        context_candidates = self.resolver.resolve(hive_id, context, use_global=use_global_memory,
                                                   max_candidates=cfg.max_seed_candidates) if context else []
        merged = self._merge_candidates(candidates, context_candidates, cfg)
        with self.repository.transaction() as conn:
            state = self._working_state(conn, hive_id)
            concepts = self._concept_states(conn, hive_id, merged, cfg)
            session = {
                "id": f"resonance-session-{uuid.uuid4().hex[:12]}",
                "hive_id": hive_id,
                "input": input_text,
                "status": "seeding" if concepts else "completed",
                "tick": 0,
                "max_ticks": cfg.max_ticks,
                "temperature": cfg.temperature,
                "energy_budget": cfg.energy_budget,
                "stability": 0.0,
                "completion_reason": "no_candidates" if not concepts else None,
                "config": asdict(cfg),
                "lexical_candidates": merged,
                "seed_concepts": [item["concept_id"] for item in concepts if item["activation"] > 0],
                "concepts": concepts,
                "active_concepts": [],
                "suppressed_concepts": [],
                "snapshots": [],
                "logs": [],
                "stable_ticks": 0,
                "save_snapshots": bool(save_snapshots),
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
            self._refresh_summary(session, cfg)
            self._snapshot(session, "seeded")
            state["resonance_sessions"] = [item for item in state.get("resonance_sessions", []) if item.get("id") != session["id"]][-19:] + [session]
            state["active_resonance_session_id"] = session["id"]
            self._save_state(conn, hive_id, state)
            return self._public(session)

    def get(self, session_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            _, session = self._find_session(conn, session_id)
            return self._public(session)

    def get_for_hive(self, hive_id: str, session_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._working_state(conn, hive_id)
            session = next((item for item in state.get("resonance_sessions", []) if item.get("id") == session_id), None)
            if not session:
                raise KeyError(session_id)
            return self._public(session)

    def step(self, session_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive_id, state, session = self._find_session(conn, session_id)
            self._step(session)
            self._persist(conn, hive_id, state, session)
            return self._public(session)

    def run(self, session_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive_id, state, session = self._find_session(conn, session_id)
            while session["status"] not in {"completed", "failed", "stopped"}:
                self._step(session)
            self._persist(conn, hive_id, state, session)
            return self._public(session)

    def stop(self, session_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive_id, state, session = self._find_session(conn, session_id)
            if session["status"] not in {"completed", "failed"}:
                session["status"] = "stopped"
                session["completion_reason"] = "stopped_by_user"
                session["updated_at"] = utcnow()
                self._snapshot(session, "stopped")
                self._persist(conn, hive_id, state, session)
            return self._public(session)

    def snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        return self.get(session_id)["snapshots"]

    def import_concept(self, session_id: str, concept_id: str) -> Dict[str, Any]:
        """Consolidate one explicitly selected global proxy into the local hive."""
        with self.repository.transaction() as conn:
            hive_id, state, session = self._find_session(conn, session_id)
            candidate = next((item for item in session["lexical_candidates"] if str(item.get("conceptId")) == str(concept_id)), None)
            if not candidate:
                raise KeyError(concept_id)
            hive = conn.execute("SELECT space_id FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            cell = self.resolver._place_seed(conn, hive_id, int(hive["space_id"]), candidate, {
                "id": session_id, "input": session["input"],
            })
            for concept in session["concepts"]:
                if concept["concept_id"] == str(concept_id):
                    concept["imported"] = True
                    concept["temporary"] = False
                    concept["source"] = "local"
            session["updated_at"] = utcnow()
            self._persist(conn, hive_id, state, session)
            return {"cell": cell, "session": self._public(session)}

    def _step(self, session: Dict[str, Any]) -> None:
        cfg = ResonanceConfig.from_values(session["config"])
        if session["status"] in {"completed", "failed", "stopped"}:
            return
        if not session["concepts"]:
            session.update(status="completed", completion_reason="no_candidates")
            return
        session["status"] = "resonating"
        concepts = session["concepts"]
        old = [float(item["activation"]) for item in concepts]
        influences = self._influences(concepts, cfg)
        raw_noise = [self._noise(session["id"], session["tick"], item["concept_id"]) for item in concepts]
        noise_mean = sum(raw_noise) / max(1, len(raw_noise))
        raw_activations: List[float] = []
        forces: List[List[float]] = []
        for index, concept in enumerate(concepts):
            incoming = influences[index]
            total_incoming = sum(item["influence"] for item in incoming if item["influence"] > 0)
            support = 1.0 - math.exp(-total_incoming)
            suppression = sum(-item["influence"] for item in incoming if item["influence"] < 0)
            noise = (raw_noise[index] - noise_mean) * cfg.temperature * .025
            value = max(0.0, min(1.0, float(concept["activation"]) * cfg.decay + support - suppression + noise))
            raw_activations.append(value)
            concept["previous_activation"] = float(concept["activation"])
            concept["received_energy"] = support
            concept["emitted_energy"] = sum(abs(item["influence"]) for item in incoming)
            concept["suppression"] = suppression
            concept["temperature_noise"] = noise
            concept["sources"] = sorted(incoming, key=lambda item: abs(item["influence"]), reverse=True)[:6]
            forces.append(self._force(incoming, concepts, index, cfg))
        self._normalize(raw_activations, cfg.energy_budget)
        position_delta = 0.0
        for index, concept in enumerate(concepts):
            concept["activation"] = raw_activations[index]
            velocity = concept["velocity"]
            force = forces[index]
            velocity = [velocity[axis] * cfg.velocity_damping + force[axis] for axis in range(2)]
            velocity = self._limit(velocity, cfg.max_velocity)
            displacement = [concept["displacement"][axis] + velocity[axis] for axis in range(2)]
            displacement = self._limit(displacement, cfg.max_displacement)
            position_delta += math.dist(displacement, concept["displacement"])
            concept["velocity"] = velocity
            concept["displacement"] = displacement
            concept["render_position"] = [concept["base_position"][axis] + displacement[axis] for axis in range(2)]
        activation_delta = sum(abs(value - old[index]) for index, value in enumerate(raw_activations)) / max(1, len(concepts))
        position_delta /= max(1, len(concepts))
        session["tick"] += 1
        session["stability"] = max(0.0, min(1.0, 1.0 - (activation_delta + position_delta) / .20))
        session["stable_ticks"] = session.get("stable_ticks", 0) + 1 if activation_delta + position_delta <= cfg.stability_threshold else 0
        self._refresh_summary(session, cfg)
        reason = None
        if not session["active_concepts"]:
            reason = "energy_dissipated"
        elif session["stable_ticks"] >= cfg.stable_ticks_required:
            reason = "stabilized"
        elif session["tick"] >= cfg.max_ticks:
            reason = "max_ticks"
        if reason:
            session["status"] = "completed"
            session["completion_reason"] = reason
        elif session["tick"] >= max(1, cfg.max_ticks - 1):
            session["status"] = "stabilizing"
        session["logs"].append({
            "tick": session["tick"], "incoming_energy": round(sum(item["received_energy"] for item in concepts), 6),
            "suppression": round(sum(item["suppression"] for item in concepts), 6),
            "temperature_noise": round(sum(item["temperature_noise"] for item in concepts), 6),
            "total_energy": round(sum(item["activation"] for item in concepts), 6),
            "activation_delta": round(activation_delta, 6), "position_delta": round(position_delta, 6),
            "stability": round(session["stability"], 6),
        })
        session["updated_at"] = utcnow()
        self._snapshot(session, "tick")

    def _influences(self, concepts: List[Dict[str, Any]], cfg: ResonanceConfig) -> List[List[Dict[str, Any]]]:
        result: List[List[Dict[str, Any]]] = [[] for _ in concepts]
        for source_index, source in enumerate(concepts):
            for target_index, target in enumerate(concepts):
                if source_index == target_index or source["activation"] <= 0:
                    continue
                distance = math.dist(source["render_position"], target["render_position"])
                overlap = max(0.0, 1.0 - distance / max(.001, source["halo_radius"] + target["halo_radius"]))
                distance_decay = math.exp(-(distance * distance) / max(.001, cfg.distance_sigma * cfg.distance_sigma))
                compatibility = self._compatibility(source, target)
                gravity = .5 + .5 * min(1.0, (source["gravity"] + target["gravity"]) / 2)
                support = source["activation"] * overlap * compatibility * gravity * distance_decay
                # Weakly compatible concepts occupying the same field compete.
                competition = source["activation"] * overlap * max(0.0, .45 - compatibility) * .55
                influence = support - competition
                if abs(influence) < .0001:
                    continue
                result[target_index].append({
                    "concept_id": source["concept_id"], "label": source["label"],
                    "influence": round(influence, 6), "support": round(support, 6),
                    "suppression": round(competition, 6), "distance": round(distance, 6),
                    "field_overlap": round(overlap, 6), "compatibility": round(compatibility, 6),
                })
        return result

    def _force(self, influences: Iterable[Dict[str, Any]], concepts: List[Dict[str, Any]],
               index: int, cfg: ResonanceConfig) -> List[float]:
        target = concepts[index]
        force = [0.0, 0.0]
        source_by_id = {item["concept_id"]: item for item in concepts}
        for influence in influences:
            source = source_by_id[influence["concept_id"]]
            direction = [source["render_position"][axis] - target["render_position"][axis] for axis in range(2)]
            length = math.hypot(*direction) or 1.0
            sign = 1.0 if influence["influence"] >= 0 else -1.0
            magnitude = min(.02, abs(influence["influence"]) * .03) * sign
            for axis in range(2):
                force[axis] += direction[axis] / length * magnitude
        return force

    @staticmethod
    def _compatibility(left: Dict[str, Any], right: Dict[str, Any]) -> float:
        if left.get("lemma") and left.get("lemma") == right.get("lemma"):
            return .92
        related_left = set(left.get("related_scene_ids", []))
        related_right = set(right.get("related_scene_ids", []))
        if related_left & related_right:
            return .82
        shared = len(set(left["label"].casefold()) & set(right["label"].casefold()))
        return min(.44, .16 + shared / max(1, len(set(left["label"]) | set(right["label"]))))

    def _concept_states(self, conn: Any, hive_id: str, candidates: List[Dict[str, Any]],
                        cfg: ResonanceConfig) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for candidate in candidates:
            score = float(candidate["lexicalScore"])
            if score < cfg.candidate_threshold:
                continue
            cloud_id = int(candidate["candidate_cloud_id"])
            row = conn.execute("""SELECT c.id,c.canonical_name,c.mass,
                hp.x AS hive_x,hp.y AS hive_y,hp.radius AS hive_radius,hc.local_activation,hc.retention,
                gp.x AS global_x,gp.y AS global_y,gp.radius AS global_radius
                FROM clouds c
                LEFT JOIN hive_cells hc ON hc.hive_id=? AND hc.dominant_cloud_id=c.id
                LEFT JOIN cloud_placements hp ON hp.id=hc.hive_placement_id
                LEFT JOIN cloud_placements gp ON gp.cloud_id=c.id AND gp.space_id=(SELECT id FROM spaces WHERE space_type='global_field' LIMIT 1)
                WHERE c.id=? ORDER BY hc.created_at DESC LIMIT 1""", (hive_id, cloud_id)).fetchone()
            if not row:
                continue
            local = row["hive_x"] is not None
            x = float(row["hive_x"] if local else row["global_x"] if row["global_x"] is not None else self.repository.stable_position("resonance", cloud_id)[0])
            y = float(row["hive_y"] if local else row["global_y"] if row["global_y"] is not None else self.repository.stable_position("resonance", cloud_id)[1])
            base_position = [x / 1600.0, y / 1000.0]
            source_weight = 1.0 if local else cfg.global_memory_weight
            context_weight = float(candidate.get("context_weight", 1.0))
            activation = score * source_weight * context_weight
            related = [str(item.get("scene_cloud_id")) for item in candidate.get("related_scenes", [])]
            result.append({
                "concept_id": str(cloud_id), "label": row["canonical_name"], "lemma": candidate.get("lemma"),
                "source": "local" if local else "global", "temporary": not local,
                "imported": local, "lexical_score": score, "matched_by": candidate["matchedBy"],
                "base_position": base_position, "render_position": list(base_position),
                "mass": float(row["mass"]), "gravity": min(1.0, float(row["retention"] or row["local_activation"] or .5)) if local else min(1.0, float(row["mass"]) / 2),
                "core_radius": .14, "halo_radius": .38,
                "activation": activation, "previous_activation": 0.0, "received_energy": 0.0,
                "emitted_energy": 0.0, "suppression": 0.0, "temperature_noise": 0.0,
                "velocity": [0.0, 0.0], "displacement": [0.0, 0.0], "sources": [],
                "related_scene_ids": related,
            })
        activations = [item["activation"] for item in result]
        self._normalize(activations, cfg.energy_budget)
        for index, value in enumerate(activations):
            result[index]["activation"] = value
        return result

    @staticmethod
    def _merge_candidates(primary: List[Dict[str, Any]], context: List[Dict[str, Any]],
                          cfg: ResonanceConfig) -> List[Dict[str, Any]]:
        merged: Dict[int, Dict[str, Any]] = {}
        for candidate in primary:
            candidate = {**candidate, "context_weight": 1.0}
            merged[int(candidate["candidate_cloud_id"])] = candidate
        for candidate in context:
            candidate = {**candidate, "context_weight": .60}
            key = int(candidate["candidate_cloud_id"])
            if key not in merged or candidate["lexicalScore"] * .60 > merged[key]["lexicalScore"]:
                merged[key] = candidate
        return sorted(merged.values(), key=lambda item: -(float(item["lexicalScore"]) * float(item["context_weight"])))[:cfg.max_seed_candidates]

    def _refresh_summary(self, session: Dict[str, Any], cfg: ResonanceConfig) -> None:
        concepts = session["concepts"]
        session["active_concepts"] = [self._concept_public(item) for item in concepts if item["activation"] >= cfg.activation_threshold]
        session["suppressed_concepts"] = [self._concept_public(item) for item in concepts if item["activation"] < cfg.activation_threshold]
        active = sorted(session["active_concepts"], key=lambda item: -item["activation"])
        session["dominant_configuration"] = ({"label": ", ".join(item["label"] for item in active[:4]),
                                               "energy": round(sum(item["activation"] for item in active[:4]), 6)} if active else None)

    def _snapshot(self, session: Dict[str, Any], event: str) -> None:
        if not session.get("save_snapshots", True):
            return
        session["snapshots"].append({
            "tick": session["tick"], "event": event, "temperature": session["temperature"],
            "total_energy": round(sum(item["activation"] for item in session["concepts"]), 6),
            "stability": round(session["stability"], 6),
            "concepts": [self._concept_public(item) for item in session["concepts"]],
            "dominant_regions": [session["dominant_configuration"]] if session.get("dominant_configuration") else [],
        })

    @staticmethod
    def _concept_public(concept: Dict[str, Any]) -> Dict[str, Any]:
        return {key: concept[key] for key in ("concept_id", "label", "source", "temporary", "imported", "lexical_score", "matched_by", "base_position", "render_position", "mass", "gravity", "core_radius", "halo_radius", "activation", "previous_activation", "received_energy", "emitted_energy", "suppression", "temperature_noise", "velocity", "displacement", "sources")}

    @staticmethod
    def _normalize(values: List[float], budget: float) -> None:
        total = sum(max(0.0, value) for value in values)
        if total > budget > 0:
            for index, value in enumerate(values):
                values[index] = max(0.0, value) * budget / total

    @staticmethod
    def _limit(vector: List[float], maximum: float) -> List[float]:
        length = math.hypot(*vector)
        return [value * maximum / length for value in vector] if length > maximum > 0 else vector

    @staticmethod
    def _noise(session_id: str, tick: int, concept_id: str) -> float:
        digest = hashlib.sha256(f"{session_id}:{tick}:{concept_id}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 2**31 - 1.0

    def _dialogue_context(self, hive_id: str, input_text: str) -> str:
        with self.repository.transaction() as conn:
            rows = conn.execute("SELECT text FROM hive_messages WHERE hive_id=? ORDER BY turn_index DESC LIMIT 4", (hive_id,)).fetchall()
        texts = [str(row["text"]) for row in reversed(rows) if str(row["text"]) != input_text]
        return " ".join(texts)

    @staticmethod
    def _working_state(conn: Any, hive_id: str) -> Dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        if not row:
            raise KeyError(hive_id)
        return decode(row["metadata_json"], {}).get("query_working_memory", {})

    @staticmethod
    def _save_state(conn: Any, hive_id: str, state: Dict[str, Any]) -> None:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        metadata = decode(row["metadata_json"], {})
        metadata["query_working_memory"] = state
        conn.execute("UPDATE hives SET metadata_json=?,updated_at=? WHERE id=?", (encode(metadata), utcnow(), hive_id))

    def _find_session(self, conn: Any, session_id: str) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        rows = conn.execute("SELECT id,metadata_json FROM hives").fetchall()
        for row in rows:
            state = decode(row["metadata_json"], {}).get("query_working_memory", {})
            session = next((item for item in state.get("resonance_sessions", []) if item.get("id") == session_id), None)
            if session:
                return str(row["id"]), state, session
        raise KeyError(session_id)

    def _persist(self, conn: Any, hive_id: str, state: Dict[str, Any], session: Dict[str, Any]) -> None:
        state["resonance_sessions"] = [session if item.get("id") == session["id"] else item for item in state.get("resonance_sessions", [])]
        state["active_resonance_session_id"] = session["id"]
        self._save_state(conn, hive_id, state)

    def _public(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {key: session.get(key) for key in ("id", "hive_id", "input", "status", "tick", "max_ticks", "temperature", "energy_budget", "stability", "completion_reason", "lexical_candidates", "seed_concepts", "active_concepts", "suppressed_concepts", "dominant_configuration", "snapshots", "logs", "created_at", "updated_at")}
