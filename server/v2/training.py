"""Idempotent training for Cloud / Space / Placement."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from server.tokenizer import SentenceTokens, WordToken, tokenize_hierarchical
from .repository import V2Repository, encode, utcnow
from .morphology import MorphologyService
from .unknown_search import StructuralIndexService


ROLE_VALUES = {
    "subject", "predicate", "object", "attribute", "location", "definition",
    "complement", "preposition", "service", "unknown",
}


@dataclass(frozen=True)
class Morphology:
    lemma: str
    pos_tag: str
    features: Dict[str, str]


class RussianMorphology:
    def __init__(self) -> None:
        try:
            import pymorphy3

            self._analyzer = pymorphy3.MorphAnalyzer()
        except Exception:
            self._analyzer = None

    def parse(self, word: str) -> Morphology:
        if not self._analyzer:
            return Morphology(word.casefold(), "UNK", {})
        parsed = self._analyzer.parse(word)[0]
        tag = parsed.tag
        features = {
            key: value
            for key, value in {
                "case": tag.case,
                "number": tag.number,
                "gender": tag.gender,
                "tense": tag.tense,
                "person": tag.person,
                "aspect": tag.aspect,
            }.items()
            if value
        }
        return Morphology(parsed.normal_form, str(tag.POS or "UNK"), features)


class RoleResolver:
    PREPOSITIONS = {"в", "во", "на", "к", "с", "со", "из", "от", "для", "по", "о", "об", "у"}
    SERVICE = {"это", "не", "ли", "бы", "же"}
    VERBS = {"VERB", "INFN", "PRTF", "PRTS"}
    NOUNS = {"NOUN", "NPRO"}
    ADJECTIVES = {"ADJF", "ADJS", "PRTF", "NUMR"}

    def resolve(self, tokens: Sequence[WordToken], morphologies: Sequence[Morphology]) -> List[Dict[str, Any]]:
        predicate_index = next((i for i, item in enumerate(morphologies) if item.pos_tag in self.VERBS), None)
        definition_index = next((i for i, token in enumerate(tokens) if token.normalized == "это"), None)
        has_source = any(
            tokens[index - 1].normalized in {"из", "от"} and morph.features.get("case") == "gent"
            for index, (token, morph) in enumerate(zip(tokens, morphologies))
            if index
        )
        result: List[Dict[str, Any]] = []
        for index, (token, morph) in enumerate(zip(tokens, morphologies)):
            previous = tokens[index - 1].normalized if index else ""
            role, dependency, confidence = "unknown", "unknown", 0.45
            if token.normalized in self.PREPOSITIONS:
                role, dependency, confidence = "preposition", "marker", 0.99
            elif token.normalized in self.SERVICE:
                role, dependency, confidence = "service", "service", 0.98
            elif previous in {"в", "во", "на"} and morph.features.get("case") in {"loct", None}:
                role, dependency, confidence = "location", "prepositional", 0.92
            elif previous in {"в", "во", "на"} and morph.features.get("case") == "accs":
                role, dependency, confidence = "destination", "prepositional", 0.92
            elif previous in {"к"} and morph.features.get("case") in {"datv", None}:
                role, dependency, confidence = "destination", "prepositional", 0.9
            elif previous in {"из", "от"} and morph.features.get("case") == "gent":
                role, dependency, confidence = "source", "prepositional", 0.92
            elif previous in {"с", "со"} and morph.features.get("case") == "gent":
                role, dependency, confidence = "source", "prepositional", 0.9
            elif previous in {"с", "со"} and morph.features.get("case") in {"ablt", None}:
                if definition_index is not None and index > definition_index:
                    role, dependency, confidence = "complement", "prepositional", 0.82
                else:
                    role, dependency, confidence = "instrument", "prepositional", 0.88
            elif definition_index is not None and index > definition_index and morph.pos_tag in self.NOUNS:
                role, dependency, confidence = "definition", "defines", 0.84
            elif morph.pos_tag in self.VERBS:
                role, dependency, confidence = "predicate", "root", 0.92
            elif morph.pos_tag in self.ADJECTIVES and index + 1 < len(tokens):
                role, dependency, confidence = "attribute", "modifies", 0.80
            elif morph.pos_tag in self.NOUNS:
                case = morph.features.get("case")
                if predicate_index is None and has_source:
                    role, dependency, confidence = "object", "theme", 0.84
                elif (predicate_index is None and index == 0) or (
                    predicate_index is not None and index < predicate_index and case in {None, "nomn"}
                ):
                    role, dependency, confidence = "subject", "subject", 0.86
                elif predicate_index is not None and (index > predicate_index or case in {"accs", "gent", "datv"}):
                    role, dependency, confidence = "object", "object", 0.78
                elif case == "ablt":
                    role, dependency, confidence = "instrument", "instrument", 0.76
            result.append({"role": role, "dependency_role": dependency, "confidence": confidence})
        return result


class TrainingJournal:
    def __init__(self, conn: Any, run_id: str) -> None:
        self.conn = conn
        self.run_id = run_id
        self.events: List[Dict[str, Any]] = []

    def record(
        self,
        event_type: str,
        entity_type: str,
        entity_id: Any,
        before: Any,
        after: Any,
        reason: str,
    ) -> None:
        event = {
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "value_before": before,
            "value_after": after,
            "reason": reason,
        }
        self.events.append(event)
        self.conn.execute(
            """INSERT INTO training_change_events
            (training_run_id, event_type, entity_type, entity_id, value_before_json,
             value_after_json, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id,
                event_type,
                entity_type,
                str(entity_id),
                encode(before) if before is not None else None,
                encode(after) if after is not None else None,
                reason,
                utcnow(),
            ),
        )


class WordFormStructureService:
    def __init__(self, repository: V2Repository) -> None:
        self.repository = repository

    def ensure_structure(
        self,
        conn: Any,
        word_cloud: Dict[str, Any],
        normalized_form: str,
        global_space_id: int,
        journal: TrainingJournal,
    ) -> Dict[str, Any]:
        structure_space, created = self.repository.get_or_create_space(
            conn,
            "word_structure_space",
            int(word_cloud["id"]),
            global_space_id,
            seed=int(hashlib.sha256(normalized_form.encode()).hexdigest()[:8], 16),
        )
        if created:
            journal.record("SPACE_CREATED", "space", structure_space["id"], None, structure_space, "word form structure")

        existing = {
            int(row["component_index"]): dict(row)
            for row in conn.execute(
                "SELECT * FROM structural_components WHERE parent_cloud_id = ?",
                (word_cloud["id"],),
            )
        }
        count = len(normalized_form)
        for index, character in enumerate(normalized_form):
            char_cloud, char_created = self.repository.get_or_create_cloud(
                conn, "character", character, stability=0.05
            )
            if char_created:
                journal.record("CLOUD_CREATED", "cloud", char_cloud["id"], None, char_cloud, "new character")
            else:
                before, char_cloud = self.repository.strengthen_cloud(conn, int(char_cloud["id"]), 0.02)
                journal.record("CLOUD_STRENGTHENED", "cloud", char_cloud["id"], before, char_cloud, "character observed")

            component = existing.get(index)
            if component:
                if int(component["child_cloud_id"]) != int(char_cloud["id"]):
                    raise ValueError(f"word structure mismatch: {normalized_form}[{index}]")
                continue
            x = 0.0 if count == 1 else -36.0 * (count - 1) / 2.0 + 36.0 * index
            cursor = conn.execute(
                """INSERT INTO structural_components
                (parent_cloud_id, child_cloud_id, component_index, component_role,
                 weight, local_x, local_y, metadata_json)
                VALUES (?, ?, ?, 'character', 1, ?, 0, '{}')""",
                (word_cloud["id"], char_cloud["id"], index, x),
            )
            component = dict(conn.execute(
                "SELECT * FROM structural_components WHERE id = ?", (cursor.lastrowid,)
            ).fetchone())
            journal.record("STRUCTURE_CREATED", "structural_component", component["id"], None, component, "missing character position")

        if len(existing) > count:
            raise ValueError(f"word structure has extra components: {normalized_form}")
        return structure_space


class TrainingPipelineV2:
    parser_version = "ru-rule-v3"

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        self.morphology = RussianMorphology()
        self.roles = RoleResolver()
        self.structures = WordFormStructureService(self.repository)
        self.morphology_space = MorphologyService(self.repository)
        self.structural_index = StructuralIndexService()

    def train(self, text: str, source_type: str = "training") -> Dict[str, Any]:
        tokenization = tokenize_hierarchical(text)
        run_id = f"train-{uuid.uuid4().hex}"
        with self.repository.transaction() as conn:
            conn.execute(
                "INSERT INTO training_runs(id, source_text, source_type, success, created_at) VALUES (?, ?, ?, 0, ?)",
                (run_id, text, source_type, utcnow()),
            )
            journal = TrainingJournal(conn, run_id)
            global_space, global_created = self.repository.get_or_create_space(conn, "global_field", seed=1337)
            if global_created:
                journal.record("SPACE_CREATED", "space", global_space["id"], None, global_space, "model root")
            self._migrate_existing_scenes(conn)
            scene_results = [
                self._train_sentence(conn, sentence, int(global_space["id"]), source_type, journal)
                for sentence in tokenization.sentences
            ]
            success = bool(scene_results)
            conn.execute(
                "UPDATE training_runs SET success = ?, completed_at = ? WHERE id = ?",
                (int(success), utcnow(), run_id),
            )
            stats = self.repository.stats(conn)

        def by_type(name: str) -> List[Dict[str, Any]]:
            return [item for item in journal.events if item["event_type"] == name]
        return {
            "success": success,
            "training_run_id": run_id,
            "scenes": scene_results,
            "created_clouds": by_type("CLOUD_CREATED"),
            "strengthened_clouds": by_type("CLOUD_STRENGTHENED"),
            "new_candidates": by_type("CANDIDATE_CREATED"),
            "activations": by_type("ACTIVATION_CHANGED"),
            "created_spaces": by_type("SPACE_CREATED"),
            "created_placements": by_type("PLACEMENT_CREATED"),
            "created_structures": by_type("STRUCTURE_CREATED"),
            "reused_scenes": by_type("SCENE_REUSED"),
            "stats": stats,
        }

    def _migrate_existing_scenes(self, conn: Any) -> None:
        rows = conn.execute(
            "SELECT cloud_id FROM scenes WHERE parser_version <> ? ORDER BY cloud_id",
            (self.parser_version,),
        ).fetchall()
        for row in rows:
            components = conn.execute(
                """SELECT sc.id, sc.token_index, wf.normalized_form
                FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
                WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""",
                (row["cloud_id"],),
            ).fetchall()
            tokens = [WordToken(text=item["normalized_form"], normalized=item["normalized_form"], position=index) for index, item in enumerate(components)]
            morphologies = [self.morphology.parse(token.normalized) for token in tokens]
            roles = self.roles.resolve(tokens, morphologies)
            predicate_id = next((int(item["id"]) for item, role in zip(components, roles) if role["role"] == "predicate"), None)
            for component, morphology, role in zip(components, morphologies, roles):
                conn.execute(
                    """UPDATE scene_components SET grammatical_role=?, dependency_role=?, confidence=?,
                    morphology_json=?, head_component_id=? WHERE id=?""",
                    (role["role"], role["dependency_role"], role["confidence"],
                     encode({"lemma": morphology.lemma, "pos": morphology.pos_tag, **morphology.features}),
                     predicate_id if predicate_id and role["role"] not in {"preposition", "service", "predicate"} else None,
                     component["id"]),
                )
            conn.execute("UPDATE scenes SET parser_version=?, updated_at=? WHERE cloud_id=?", (self.parser_version, utcnow(), row["cloud_id"]))

    def _cloud(
        self,
        conn: Any,
        cloud_type: str,
        name: str,
        journal: TrainingJournal,
        **defaults: Any,
    ) -> Tuple[Dict[str, Any], bool]:
        cloud, created = self.repository.get_or_create_cloud(conn, cloud_type, name, **defaults)
        if created:
            journal.record("CLOUD_CREATED", "cloud", cloud["id"], None, cloud, f"new {cloud_type}")
        else:
            before, cloud = self.repository.strengthen_cloud(conn, int(cloud["id"]))
            journal.record("CLOUD_STRENGTHENED", "cloud", cloud["id"], before, cloud, f"observed {cloud_type}")
        return cloud, created

    def _global_placement(
        self,
        conn: Any,
        cloud: Dict[str, Any],
        global_space_id: int,
        journal: TrainingJournal,
    ) -> Dict[str, Any]:
        placement, created = self.repository.ensure_global_placement(conn, cloud, global_space_id)
        if created:
            journal.record("PLACEMENT_CREATED", "placement", placement["id"], None, placement, "global projection")
        return placement

    def _train_sentence(
        self,
        conn: Any,
        sentence: SentenceTokens,
        global_space_id: int,
        source_type: str,
        journal: TrainingJournal,
    ) -> Dict[str, Any]:
        canonical = " ".join(token.normalized for token in sentence.tokens)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        scene, scene_created = self._cloud(
            conn, "scene", canonical, journal, mass=2.0, density=0.85, stability=0.4
        )
        scene_space, space_created = self.repository.get_or_create_space(
            conn,
            "scene_space",
            int(scene["id"]),
            global_space_id,
            seed=int(fingerprint[:8], 16),
        )
        if space_created:
            journal.record("SPACE_CREATED", "space", scene_space["id"], None, scene_space, "scene local space")
        self._global_placement(conn, scene, global_space_id, journal)

        scene_row = conn.execute("SELECT * FROM scenes WHERE cloud_id = ?", (scene["id"],)).fetchone()
        if scene_row:
            before = dict(scene_row)
            conn.execute(
                "UPDATE scenes SET observation_count = observation_count + 1, updated_at = ? WHERE cloud_id = ?",
                (utcnow(), scene["id"]),
            )
            after = dict(conn.execute("SELECT * FROM scenes WHERE cloud_id = ?", (scene["id"],)).fetchone())
            journal.record("SCENE_REUSED", "scene", scene["id"], before, after, "same normalized sentence")
        else:
            now = utcnow()
            conn.execute(
                """INSERT INTO scenes
                (cloud_id, scene_space_id, sentence_text, canonical_text, fingerprint,
                 parser_version, observation_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (scene["id"], scene_space["id"], sentence.text, canonical, fingerprint, self.parser_version, now, now),
            )

        morphologies = [self.morphology.parse(token.normalized) for token in sentence.tokens]
        roles = self.roles.resolve(sentence.tokens, morphologies)
        token_data: List[Tuple[Dict[str, Any], Dict[str, Any], Morphology, Dict[str, Any]]] = []
        for token, morphology, role in zip(sentence.tokens, morphologies, roles):
            word, lexeme = self._ensure_word(conn, token, morphology, global_space_id, journal)
            token_data.append((word, lexeme, morphology, role))

        if not scene_created and scene_row:
            component_count = conn.execute(
                "SELECT COUNT(*) FROM scene_components WHERE scene_cloud_id = ?", (scene["id"],)
            ).fetchone()[0]
            if int(component_count) != len(sentence.tokens):
                raise ValueError("reused scene has invalid component count")
            existing_components = conn.execute(
                "SELECT id, token_index FROM scene_components WHERE scene_cloud_id=? ORDER BY token_index",
                (scene["id"],),
            ).fetchall()
            component_ids = [int(row["id"]) for row in existing_components]
            for index, (word, lexeme, morphology, role) in enumerate(token_data):
                conn.execute(
                    """UPDATE scene_components
                    SET word_form_cloud_id=?, lexeme_cloud_id=?, grammatical_role=?,
                        dependency_role=?, confidence=?, morphology_json=?
                    WHERE scene_cloud_id=? AND token_index=?""",
                    (word["id"], lexeme["id"], role["role"], role["dependency_role"],
                     role["confidence"], encode({"lemma": morphology.lemma, "pos": morphology.pos_tag, **morphology.features}),
                     scene["id"], index),
                )
            predicate = next((component_ids[i] for i, item in enumerate(roles) if item["role"] == "predicate"), None)
            if predicate:
                for component_id, role in zip(component_ids, roles):
                    if component_id != predicate and role["role"] not in {"preposition", "service"}:
                        conn.execute("UPDATE scene_components SET head_component_id=? WHERE id=?", (predicate, component_id))
            conn.execute("UPDATE scenes SET parser_version=?, updated_at=? WHERE cloud_id=?", (self.parser_version, utcnow(), scene["id"]))
        else:
            component_ids: List[int] = []
            for index, (word, lexeme, morphology, role) in enumerate(token_data):
                x = 160.0 + index * 140.0
                y = 260.0
                placement = self.repository.create_placement(
                    conn,
                    int(word["id"]),
                    int(scene_space["id"]),
                    x,
                    y,
                    local_activation=1.0,
                    metadata={"placement_kind": "scene_component", "scene_cloud_id": scene["id"], "token_index": index},
                )
                journal.record("PLACEMENT_CREATED", "placement", placement["id"], None, placement, "word occurrence in scene")
                journal.record(
                    "ACTIVATION_CHANGED",
                    "placement",
                    placement["id"],
                    {"local_activation": 0.0},
                    {"local_activation": 1.0},
                    "token observed in scene",
                )
                cursor = conn.execute(
                    """INSERT INTO scene_components
                    (scene_cloud_id, word_form_cloud_id, lexeme_cloud_id, placement_id,
                     token_index, grammatical_role, dependency_role, confidence, morphology_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        scene["id"],
                        word["id"],
                        lexeme["id"],
                        placement["id"],
                        index,
                        role["role"],
                        role["dependency_role"],
                        role["confidence"],
                        encode({"lemma": morphology.lemma, "pos": morphology.pos_tag, **morphology.features}),
                    ),
                )
                component_ids.append(int(cursor.lastrowid))
            predicate = next((component_ids[i] for i, item in enumerate(roles) if item["role"] == "predicate"), None)
            if predicate:
                for component_id, role in zip(component_ids, roles):
                    if component_id != predicate and role["role"] not in {"preposition", "service"}:
                        conn.execute(
                            "UPDATE scene_components SET head_component_id = ? WHERE id = ?",
                            (predicate, component_id),
                        )

        conn.execute(
            """INSERT INTO training_observations
            (training_run_id, source_text, normalized_text, scene_cloud_id, source_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (journal.run_id, sentence.text, canonical, scene["id"], source_type, utcnow()),
        )
        return {"scene_cloud_id": int(scene["id"]), "created": scene_created, "canonical_text": canonical}

    def _ensure_word(
        self,
        conn: Any,
        token: WordToken,
        morphology: Morphology,
        global_space_id: int,
        journal: TrainingJournal,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        lexeme, lexeme_created = self._cloud(
            conn, "lexeme", morphology.lemma, journal, stability=0.15
        )
        self._global_placement(conn, lexeme, global_space_id, journal)
        conn.execute(
            """INSERT INTO lexemes(cloud_id, lemma, language, pos_tag, frequency, semantic_state)
            VALUES (?, ?, 'ru', ?, 1, 'candidate')
            ON CONFLICT(cloud_id) DO UPDATE SET
              pos_tag = excluded.pos_tag, frequency = lexemes.frequency + 1""",
            (lexeme["id"], morphology.lemma, morphology.pos_tag),
        )

        word, _ = self._cloud(conn, "word_form", token.normalized, journal, stability=0.12)
        self._global_placement(conn, word, global_space_id, journal)
        morphology_json = encode({"lemma": morphology.lemma, "pos": morphology.pos_tag, **morphology.features})
        previous = conn.execute("SELECT lexeme_cloud_id FROM word_forms WHERE cloud_id = ?", (word["id"],)).fetchone()
        conn.execute(
            """INSERT INTO word_forms
            (cloud_id, normalized_form, language, lexeme_cloud_id, pos_tag, morphology_json)
            VALUES (?, ?, 'ru', ?, ?, ?)
            ON CONFLICT(cloud_id) DO UPDATE SET
              lexeme_cloud_id = excluded.lexeme_cloud_id,
              pos_tag = excluded.pos_tag,
              morphology_json = excluded.morphology_json""",
            (word["id"], token.normalized, lexeme["id"], morphology.pos_tag, morphology_json),
        )
        if previous is None or previous["lexeme_cloud_id"] != lexeme["id"]:
            journal.record(
                "LEXEME_LINKED",
                "word_form",
                word["id"],
                {"lexeme_cloud_id": previous["lexeme_cloud_id"]} if previous else None,
                {"lexeme_cloud_id": lexeme["id"]},
                "morphological normalization",
            )

        self.structures.ensure_structure(conn, word, token.normalized, global_space_id, journal)
        self.structural_index.record(conn, int(word["id"]), token.normalized)
        self.structural_index.record(conn, int(lexeme["id"]), morphology.lemma)
        self.morphology_space.record_form(
            conn, int(word["id"]), int(lexeme["id"]),
            {"lemma": morphology.lemma, "pos": morphology.pos_tag, **morphology.features},
        )
        self.morphology_space.learn_differences(conn, int(lexeme["id"]))
        candidate, candidate_created = self._cloud(
            conn,
            "concept_candidate",
            morphology.lemma,
            journal,
            mass=0.5,
            density=0.6,
            stability=0.05,
        )
        self._global_placement(conn, candidate, global_space_id, journal)
        if candidate_created:
            journal.record("CANDIDATE_CREATED", "cloud", candidate["id"], None, candidate, "new lexeme candidate")
        conn.execute(
            """INSERT INTO semantic_memberships
            (lexeme_cloud_id, concept_cloud_id, weight, confidence, evidence_count, updated_at)
            VALUES (?, ?, 1, 0.5, 1, ?)
            ON CONFLICT(lexeme_cloud_id, concept_cloud_id) DO UPDATE SET
              evidence_count = semantic_memberships.evidence_count + 1,
              confidence = MIN(1, semantic_memberships.confidence + 0.02),
              updated_at = excluded.updated_at""",
            (lexeme["id"], candidate["id"], utcnow()),
        )
        return word, lexeme
