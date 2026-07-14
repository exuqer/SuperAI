"""Russian lexemes, distributional context vectors and fuzzy concepts."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymorphy3 import MorphAnalyzer

from server.database import get_connection, now


_morph = MorphAnalyzer(lang="ru")


@dataclass
class Lexeme:
    id: Optional[int] = None
    canonical_form: str = ""
    language: str = "ru"
    pos_tag: Optional[str] = None
    features_json: str = "{}"
    frequency: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def features(self) -> Dict[str, Any]:
        return json.loads(self.features_json or "{}")

    @features.setter
    def features(self, value: Dict[str, Any]) -> None:
        self.features_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canonical_form": self.canonical_form,
            "language": self.language,
            "pos_tag": self.pos_tag,
            "features": self.features,
            "frequency": self.frequency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LexemeService:
    def get_layer_id(self, layer_name: str) -> Optional[int]:
        with get_connection() as conn:
            row = conn.execute("SELECT id FROM layers WHERE name = ?", (layer_name,)).fetchone()
        return int(row["id"]) if row else None

    def normalize_russian(self, word: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
        parsed = _morph.parse(word.casefold())
        if not parsed:
            return word.casefold(), None, {}
        best = parsed[0]
        return best.normal_form, best.tag.POS, {
            "tag": str(best.tag),
            "score": float(best.score),
            "methods": [str(method) for method in best.methods_stack],
        }

    def normalize_unknown(self, word: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
        return word.casefold(), None, {"method": "casefold"}

    def get_or_create_lexeme(self, word_form: str, language: str = "ru") -> Lexeme:
        canonical, pos, features = (
            self.normalize_russian(word_form)
            if language == "ru"
            else self.normalize_unknown(word_form)
        )
        timestamp = now()
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM lexemes WHERE canonical_form = ? AND language = ?",
                (canonical, language),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE lexemes SET frequency = frequency + 1, updated_at = ? WHERE id = ?",
                    (timestamp, row["id"]),
                )
                conn.commit()
                data = dict(row)
                data["frequency"] += 1
                data["updated_at"] = timestamp
                return self._row_to_lexeme(data)

            cursor = conn.execute(
                """INSERT INTO lexemes
                (canonical_form, language, pos_tag, features_json, frequency, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (
                    canonical,
                    language,
                    pos,
                    json.dumps(features, ensure_ascii=False, separators=(",", ":")),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
            return Lexeme(
                id=int(cursor.lastrowid),
                canonical_form=canonical,
                language=language,
                pos_tag=pos,
                features_json=json.dumps(features, ensure_ascii=False, separators=(",", ":")),
                frequency=1,
                created_at=datetime.fromisoformat(timestamp),
                updated_at=datetime.fromisoformat(timestamp),
            )

    def get_lexeme_by_id(self, lexeme_id: int) -> Optional[Lexeme]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM lexemes WHERE id = ?", (lexeme_id,)).fetchone()
        return self._row_to_lexeme(row) if row else None

    def link_word_form_to_lexeme(
        self, word_form_cloud_id: int, lexeme_id: int, is_canonical: bool = False
    ) -> None:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO word_form_to_lexeme
                (word_form_cloud_id, lexeme_id, is_canonical, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(word_form_cloud_id, lexeme_id) DO UPDATE SET
                    is_canonical = MAX(is_canonical, excluded.is_canonical)""",
                (word_form_cloud_id, lexeme_id, int(is_canonical), now()),
            )
            conn.commit()

    def get_lexeme_for_word_form(self, word_form_cloud_id: int) -> Optional[Lexeme]:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT l.* FROM lexemes l
                JOIN word_form_to_lexeme w ON w.lexeme_id = l.id
                WHERE w.word_form_cloud_id = ?
                ORDER BY w.is_canonical DESC LIMIT 1""",
                (word_form_cloud_id,),
            ).fetchone()
        return self._row_to_lexeme(row) if row else None

    def get_word_forms_for_lexeme(self, lexeme_id: int) -> List[int]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT word_form_cloud_id FROM word_form_to_lexeme WHERE lexeme_id = ?",
                (lexeme_id,),
            ).fetchall()
        return [int(row["word_form_cloud_id"]) for row in rows]

    def accumulate_context(self, sentence_lexeme_ids: List[int], window_size: int = 5) -> None:
        if len(sentence_lexeme_ids) < 2:
            return

        increments: Counter[Tuple[int, int, int]] = Counter()
        counts: Counter[Tuple[int, int, int]] = Counter()
        for target_index, target_id in enumerate(sentence_lexeme_ids):
            start = max(0, target_index - window_size)
            end = min(len(sentence_lexeme_ids), target_index + window_size + 1)
            for context_index in range(start, end):
                if target_index == context_index:
                    continue
                context_id = sentence_lexeme_ids[context_index]
                distance = abs(target_index - context_index)
                direction = -1 if context_index < target_index else 1
                key = (target_id, context_id, direction)
                increments[key] += 1.0 / distance
                counts[key] += 1

        timestamp = now()
        with get_connection() as conn:
            for (lexeme_id, context_id, direction), raw_delta in increments.items():
                conn.execute(
                    """INSERT INTO context_vectors
                    (lexeme_id, context_lexeme_id, direction, weight, raw_weight, count, updated_at)
                    VALUES (?, ?, ?, 0, ?, ?, ?)
                    ON CONFLICT(lexeme_id, context_lexeme_id, direction) DO UPDATE SET
                        raw_weight = raw_weight + excluded.raw_weight,
                        count = count + excluded.count,
                        updated_at = excluded.updated_at""",
                    (
                        lexeme_id,
                        context_id,
                        direction,
                        raw_delta,
                        counts[(lexeme_id, context_id, direction)],
                        timestamp,
                    ),
                )
            self._recompute_ppmi(conn)
            conn.commit()

    def _recompute_ppmi(self, conn=None) -> None:
        owns_connection = conn is None
        context = get_connection() if owns_connection else None
        if owns_connection:
            conn = context.__enter__()
        try:
            rows = conn.execute(
                """SELECT lexeme_id, context_lexeme_id, direction, raw_weight AS raw
                FROM context_vectors WHERE lexeme_id != context_lexeme_id"""
            ).fetchall()
            if not rows:
                return
            row_totals: Counter[int] = Counter()
            column_totals: Counter[Tuple[int, int]] = Counter()
            total = 0.0
            for row in rows:
                raw = max(0.0, float(row["raw"]))
                row_totals[int(row["lexeme_id"])] += raw
                column_totals[(int(row["context_lexeme_id"]), int(row["direction"]))] += raw
                total += raw
            if total <= 0:
                return
            timestamp = now()
            for row in rows:
                source = int(row["lexeme_id"])
                target = int(row["context_lexeme_id"])
                direction = int(row["direction"])
                raw = max(0.0, float(row["raw"]))
                denominator = row_totals[source] * column_totals[(target, direction)]
                ppmi = max(0.0, math.log((raw * total) / denominator)) if raw and denominator else 0.0
                conn.execute(
                    """UPDATE context_vectors SET weight = ?, updated_at = ?
                    WHERE lexeme_id = ? AND context_lexeme_id = ? AND direction = ?""",
                    (ppmi, timestamp, source, target, direction),
                )
            if owns_connection:
                conn.commit()
        finally:
            if owns_connection:
                context.__exit__(None, None, None)

    def get_context_vector(self, lexeme_id: int, top_k: int = 100) -> Dict[int, float]:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT context_lexeme_id, direction, weight FROM context_vectors
                WHERE lexeme_id = ? AND context_lexeme_id != lexeme_id AND weight > 0
                ORDER BY weight DESC LIMIT ?""",
                (lexeme_id, top_k),
            ).fetchall()
        return {
            int(row["context_lexeme_id"]) * 2 + (1 if int(row["direction"]) > 0 else 0): float(row["weight"])
            for row in rows
        }

    @staticmethod
    def cosine_similarity(vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
        normalized_a = {int(key): float(value) for key, value in vec_a.items()}
        normalized_b = {int(key): float(value) for key, value in vec_b.items()}
        common = normalized_a.keys() & normalized_b.keys()
        if not common:
            return 0.0
        norm_a = math.sqrt(sum(value * value for value in normalized_a.values()))
        norm_b = math.sqrt(sum(value * value for value in normalized_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return sum(normalized_a[key] * normalized_b[key] for key in common) / (norm_a * norm_b)

    def discover_concepts(
        self,
        min_contexts: int = 3,
        similarity_threshold: float = 0.72,
    ) -> List[int]:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT l.id, l.pos_tag, l.features_json, COUNT(cv.context_lexeme_id) AS contexts
                FROM lexemes l
                JOIN context_vectors cv ON cv.lexeme_id = l.id
                WHERE cv.context_lexeme_id != l.id AND cv.weight > 0
                GROUP BY l.id HAVING contexts >= ?""",
                (min_contexts,),
            ).fetchall()
        lexeme_ids = [int(row["id"]) for row in rows]
        signatures = {int(row["id"]): row["pos_tag"] or "" for row in rows}
        vectors = {lexeme_id: self.get_context_vector(lexeme_id) for lexeme_id in lexeme_ids}
        adjacency: Dict[int, set[int]] = {lexeme_id: set() for lexeme_id in lexeme_ids}
        for index, left in enumerate(lexeme_ids):
            for right in lexeme_ids[index + 1 :]:
                if (
                    signatures[left] == signatures[right]
                    and self.cosine_similarity(vectors[left], vectors[right]) >= similarity_threshold
                ):
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        created: List[int] = []
        visited: set[int] = set()
        for seed in lexeme_ids:
            if seed in visited or not adjacency[seed]:
                continue
            component: set[int] = set()
            stack = [seed]
            while stack:
                item = stack.pop()
                if item in component:
                    continue
                component.add(item)
                stack.extend(adjacency[item] - component)
            visited.update(component)
            concept_id = self.find_or_create_concept(
                sorted(component),
                min_contexts=min_contexts,
                similarity_threshold=similarity_threshold,
            )
            if concept_id is not None:
                created.append(concept_id)
        return created

    def find_or_create_concept(
        self,
        lexeme_ids: List[int],
        min_contexts: int = 3,
        similarity_threshold: float = 0.72,
        merge_threshold: float = 0.85,
        unify_threshold: float = 0.92,
    ) -> Optional[int]:
        unique_ids = list(dict.fromkeys(int(item) for item in lexeme_ids))
        vectors = {item: self.get_context_vector(item) for item in unique_ids}
        vectors = {item: vector for item, vector in vectors.items() if len(vector) >= min_contexts}
        if len(vectors) < 2:
            return None
        pairwise = [
            self.cosine_similarity(vectors[left], vectors[right])
            for index, left in enumerate(vectors)
            for right in list(vectors)[index + 1 :]
        ]
        if not pairwise or max(pairwise) < similarity_threshold:
            return None

        centroid = self._centroid(vectors.values())
        best_id: Optional[int] = None
        best_similarity = 0.0
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT concept_cloud_id, centroid_vector_json FROM concept_centroids"
            ).fetchall()
        for row in rows:
            existing = {int(key): float(value) for key, value in json.loads(row["centroid_vector_json"]).items()}
            similarity = self.cosine_similarity(centroid, existing)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = int(row["concept_cloud_id"])
        if best_id is not None and best_similarity >= similarity_threshold:
            self._add_lexemes_to_concept(best_id, list(vectors))
            return best_id
        return self._create_concept_from_lexemes(list(vectors))

    def _create_concept_from_lexemes(self, lexeme_ids: List[int]) -> Optional[int]:
        vectors = {item: self.get_context_vector(item) for item in lexeme_ids}
        vectors = {item: vector for item, vector in vectors.items() if vector}
        if len(vectors) < 2:
            return None
        centroid = self._centroid(vectors.values())
        with get_connection() as conn:
            placeholders = ",".join("?" for _ in vectors)
            name_rows = conn.execute(
                f"SELECT id, canonical_form FROM lexemes WHERE id IN ({placeholders})",
                tuple(vectors),
            ).fetchall()
            names = {int(row["id"]): row["canonical_form"] for row in name_rows}
            ordered_ids = sorted(vectors, key=lambda item: (-self.cosine_similarity(vectors[item], centroid), names.get(item, "")))
            concept_name = " · ".join(names.get(item, str(item)) for item in ordered_ids[:3])
            layer = conn.execute("SELECT id FROM layers WHERE name = 'concept'").fetchone()
            if not layer:
                return None
            existing = conn.execute(
                "SELECT id FROM clouds WHERE layer_id = ? AND canonical_name = ?",
                (layer["id"], concept_name),
            ).fetchone()
            if existing:
                concept_id = int(existing["id"])
            else:
                timestamp = now()
                cursor = conn.execute(
                    """INSERT INTO clouds
                    (layer_id, cloud_type, canonical_name, mass, density, radius, stability,
                     activation, observation_count, created_at, updated_at, metadata_json)
                    VALUES (?, 'concept', ?, ?, 1, ?, 0.35, 0, 1, ?, ?, '{}')""",
                    (layer["id"], concept_name, max(2.0, float(len(vectors))), 42.0 + 5.0 * len(vectors), timestamp, timestamp),
                )
                concept_id = int(cursor.lastrowid)
            centroid_row = conn.execute(
                "SELECT id FROM concept_centroids WHERE concept_cloud_id = ?", (concept_id,)
            ).fetchone()
            centroid_json = json.dumps(centroid, separators=(",", ":"))
            members_json = json.dumps(ordered_ids, separators=(",", ":"))
            if centroid_row:
                conn.execute(
                    """UPDATE concept_centroids SET centroid_vector_json = ?,
                    member_lexeme_ids_json = ?, updated_at = ? WHERE id = ?""",
                    (centroid_json, members_json, now(), centroid_row["id"]),
                )
            else:
                conn.execute(
                    """INSERT INTO concept_centroids
                    (concept_cloud_id, centroid_vector_json, member_lexeme_ids_json, stability, created_at, updated_at)
                    VALUES (?, ?, ?, 0.5, ?, ?)""",
                    (concept_id, centroid_json, members_json, now(), now()),
                )
            self._write_memberships(conn, concept_id, vectors, centroid)
            conn.commit()
        return concept_id

    def _add_lexemes_to_concept(self, concept_cloud_id: int, lexeme_ids: List[int]) -> None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT member_lexeme_ids_json FROM concept_centroids WHERE concept_cloud_id = ?",
                (concept_cloud_id,),
            ).fetchone()
        if not row:
            return
        members = sorted(set(json.loads(row["member_lexeme_ids_json"] or "[]")) | set(lexeme_ids))
        vectors = {item: self.get_context_vector(item) for item in members}
        vectors = {item: vector for item, vector in vectors.items() if vector}
        if len(vectors) < 2:
            return
        centroid = self._centroid(vectors.values())
        with get_connection() as conn:
            conn.execute(
                """UPDATE concept_centroids SET centroid_vector_json = ?,
                member_lexeme_ids_json = ?, stability = MIN(1, stability + 0.03), updated_at = ?
                WHERE concept_cloud_id = ?""",
                (
                    json.dumps(centroid, separators=(",", ":")),
                    json.dumps(sorted(vectors), separators=(",", ":")),
                    now(),
                    concept_cloud_id,
                ),
            )
            self._write_memberships(conn, concept_cloud_id, vectors, centroid)
            conn.execute(
                """UPDATE clouds SET mass = MAX(mass, ?), observation_count = observation_count + 1,
                stability = MIN(1, stability + 0.02), updated_at = ? WHERE id = ?""",
                (float(len(vectors)), now(), concept_cloud_id),
            )
            conn.commit()

    def _write_memberships(
        self,
        conn,
        concept_cloud_id: int,
        vectors: Dict[int, Dict[int, float]],
        centroid: Dict[int, float],
    ) -> None:
        for lexeme_id, vector in vectors.items():
            similarity = self.cosine_similarity(vector, centroid)
            membership = self._smoothstep(0.55, 0.85, similarity)
            conn.execute(
                """INSERT INTO lexeme_concept_membership
                (lexeme_id, concept_cloud_id, membership, centrality, context_coverage, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(lexeme_id, concept_cloud_id) DO UPDATE SET
                    membership = excluded.membership,
                    centrality = excluded.centrality,
                    context_coverage = excluded.context_coverage,
                    updated_at = excluded.updated_at""",
                (
                    lexeme_id,
                    concept_cloud_id,
                    membership,
                    similarity,
                    min(1.0, len(vector) / max(1, len(centroid))),
                    now(),
                ),
            )

    def get_concept_members(self, concept_cloud_id: int) -> List[Tuple[int, float]]:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT lexeme_id, membership FROM lexeme_concept_membership
                WHERE concept_cloud_id = ? AND membership > 0 ORDER BY membership DESC""",
                (concept_cloud_id,),
            ).fetchall()
        return [(int(row["lexeme_id"]), float(row["membership"])) for row in rows]

    def merge_concepts(self, concept_a_id: int, concept_b_id: int, threshold: float = 0.92) -> Optional[int]:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT concept_cloud_id, centroid_vector_json, member_lexeme_ids_json
                FROM concept_centroids WHERE concept_cloud_id IN (?, ?)""",
                (concept_a_id, concept_b_id),
            ).fetchall()
        data = {int(row["concept_cloud_id"]): row for row in rows}
        if concept_a_id not in data or concept_b_id not in data:
            return None
        left = json.loads(data[concept_a_id]["centroid_vector_json"])
        right = json.loads(data[concept_b_id]["centroid_vector_json"])
        if self.cosine_similarity(left, right) < threshold:
            return None
        members = sorted(
            set(json.loads(data[concept_a_id]["member_lexeme_ids_json"]))
            | set(json.loads(data[concept_b_id]["member_lexeme_ids_json"]))
        )
        self._add_lexemes_to_concept(concept_a_id, members)
        with get_connection() as conn:
            conn.execute("DELETE FROM lexeme_concept_membership WHERE concept_cloud_id = ?", (concept_b_id,))
            conn.execute("DELETE FROM concept_centroids WHERE concept_cloud_id = ?", (concept_b_id,))
            conn.execute("DELETE FROM clouds WHERE id = ?", (concept_b_id,))
            conn.commit()
        return concept_a_id

    @staticmethod
    def _centroid(vectors: Iterable[Dict[int, float]]) -> Dict[int, float]:
        items = list(vectors)
        keys = set().union(*(vector.keys() for vector in items))
        return {int(key): sum(vector.get(key, 0.0) for vector in items) / len(items) for key in keys}

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, value: float) -> float:
        position = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
        return position * position * (3.0 - 2.0 * position)

    @staticmethod
    def _row_to_lexeme(row) -> Lexeme:
        return Lexeme(
            id=int(row["id"]),
            canonical_form=row["canonical_form"],
            language=row["language"],
            pos_tag=row["pos_tag"],
            features_json=row["features_json"] or "{}",
            frequency=int(row["frequency"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )


lexeme_service = LexemeService()
