"""Idempotent V2 training pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from server.tokenizer import SentenceTokens, WordToken, normalize_text, tokenize_hierarchical
from .repository import V2Repository, encode, utcnow


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
            key: value for key, value in {
                "case": tag.case, "number": tag.number, "gender": tag.gender,
                "tense": tag.tense, "person": tag.person, "aspect": tag.aspect,
            }.items() if value
        }
        return Morphology(parsed.normal_form, str(tag.POS or "UNK"), features)


class RoleResolver:
    """Small deterministic parser for the supported Russian sentence forms."""

    PREPOSITIONS = {"в", "во", "на", "к", "с", "со", "из", "от", "для", "по", "о", "об", "у"}
    SERVICE = {"это", "не", "ли", "бы"}
    VERBS = {"VERB", "INFN", "PRTF", "PRTS"}
    NOUNS = {"NOUN", "NPRO"}
    ADJECTIVES = {"ADJF", "ADJS", "PRTF", "NUMR"}

    def resolve(self, tokens: Sequence[WordToken], morphologies: Sequence[Morphology]) -> List[Dict[str, Any]]:
        predicate_index = next((i for i, m in enumerate(morphologies) if m.pos_tag in self.VERBS), None)
        definition_index = next((i for i, token in enumerate(tokens) if token.normalized == "это"), None)
        result: List[Dict[str, Any]] = []
        for index, (token, morph) in enumerate(zip(tokens, morphologies)):
            previous = tokens[index - 1].normalized if index else ""
            role, dependency, confidence = "unknown", "unknown", 0.45
            if token.normalized in self.PREPOSITIONS:
                role, dependency, confidence = "preposition", "marker", 0.99
            elif token.normalized in self.SERVICE:
                role, dependency, confidence = "service", "service", 0.98
            elif definition_index is not None and index > definition_index and morph.pos_tag in self.NOUNS:
                role, dependency, confidence = "definition", "defines", 0.84
            elif morph.pos_tag in self.VERBS:
                role, dependency, confidence = "predicate", "root", 0.92
            elif previous in {"в", "во", "на"}:
                role, dependency, confidence = "location", "prepositional", 0.88
            elif previous in {"с", "со"}:
                role, dependency, confidence = "complement", "prepositional", 0.82
            elif morph.pos_tag in self.ADJECTIVES and index + 1 < len(tokens):
                role, dependency, confidence = "attribute", "modifies", 0.8
            elif morph.pos_tag in self.NOUNS and predicate_index is not None:
                case = morph.features.get("case")
                if index < predicate_index and case in {None, "nomn"}:
                    role, dependency, confidence = "subject", "subject", 0.86
                elif case in {"accs", "gent", "datv", "ablt"} or index > predicate_index:
                    role, dependency, confidence = "object", "object", 0.78
            result.append({"role": role, "dependency_role": dependency, "confidence": confidence})
        return result


class TrainingPipelineV2:
    parser_version = "ru-rule-v1"

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        self.morphology = RussianMorphology()
        self.roles = RoleResolver()

    def train(self, text: str, source_type: str = "training") -> Dict[str, Any]:
        tokenization = tokenize_hierarchical(text)
        created: List[int] = []
        strengthened: List[int] = []
        with self.repository.transaction() as conn:
            global_space = self.repository.ensure_space(conn, "global_field", seed=1337)
            for sentence in tokenization.sentences:
                outcome = self._train_sentence(conn, sentence, int(global_space["id"]), source_type)
                (created if outcome["created"] else strengthened).append(outcome["scene_cloud_id"])
        return {
            "success": bool(tokenization.sentences), "created_scene_cloud_ids": created,
            "strengthened_scene_cloud_ids": strengthened, "tokens": len(tokenization.all_tokens),
        }

    def _train_sentence(self, conn: Any, sentence: SentenceTokens, global_space_id: int, source_type: str) -> Dict[str, Any]:
        canonical = " ".join(token.normalized for token in sentence.tokens)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        scene, scene_created = self.repository.get_or_create_cloud(
            conn, "scene", canonical, mass=2.0, density=0.85, stability=0.4
        )
        scene_space = self.repository.ensure_space(
            conn, "scene_space", int(scene["id"]), global_space_id, seed=int(fingerprint[:8], 16)
        )
        scene_row = conn.execute("SELECT * FROM v2_scenes WHERE cloud_id = ?", (scene["id"],)).fetchone()
        if scene_row:
            conn.execute(
                "UPDATE v2_scenes SET observation_count = observation_count + 1, updated_at = ? WHERE cloud_id = ?",
                (utcnow(), scene["id"]),
            )
            conn.execute(
                "INSERT INTO v2_training_observations(source_text, normalized_text, scene_cloud_id, source_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (sentence.text, canonical, scene["id"], source_type, utcnow()),
            )
            return {"scene_cloud_id": int(scene["id"]), "created": False}

        self.repository.ensure_global_placement(conn, scene, global_space_id)
        conn.execute(
            """INSERT INTO v2_scenes(cloud_id, scene_space_id, sentence_text, canonical_text, fingerprint,
            observation_count, parser_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (scene["id"], scene_space["id"], sentence.text, canonical, fingerprint,
             self.parser_version, utcnow(), utcnow()),
        )
        morphologies = [self.morphology.parse(token.normalized) for token in sentence.tokens]
        roles = self.roles.resolve(sentence.tokens, morphologies)
        component_ids: List[int] = []
        for index, (token, morph, role) in enumerate(zip(sentence.tokens, morphologies, roles)):
            word_cloud, lexeme_cloud = self._ensure_word(conn, token, morph, global_space_id)
            x = 90.0 + index * 140.0
            placement = self.repository.create_placement(
                conn, int(word_cloud["id"]), int(scene_space["id"]), x, 220.0,
                local_activation=1.0, metadata={"scene_cloud_id": scene["id"], "token_index": index},
            )
            self.repository.add_component(conn, int(scene["id"]), int(word_cloud["id"]), index, role["role"], x, 220.0)
            cursor = conn.execute(
                """INSERT INTO v2_scene_components(scene_cloud_id, word_form_cloud_id, lexeme_cloud_id,
                placement_id, token_index, grammatical_role, dependency_role, confidence, morphology_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (scene["id"], word_cloud["id"], lexeme_cloud["id"], placement["id"], index,
                 role["role"], role["dependency_role"], role["confidence"],
                 encode({"lemma": morph.lemma, "pos": morph.pos_tag, **morph.features})),
            )
            component_ids.append(cursor.lastrowid)
            self._ensure_candidate(conn, lexeme_cloud, global_space_id)
        for index, component_id in enumerate(component_ids):
            if index:
                conn.execute("UPDATE v2_scene_components SET head_component_id = ? WHERE id = ?", (component_ids[index - 1], component_id))
        conn.execute(
            "INSERT INTO v2_training_observations(source_text, normalized_text, scene_cloud_id, source_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (sentence.text, canonical, scene["id"], source_type, utcnow()),
        )
        return {"scene_cloud_id": int(scene["id"]), "created": scene_created}

    def _ensure_word(self, conn: Any, token: WordToken, morph: Morphology, global_space_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        lexeme, _ = self.repository.get_or_create_cloud(conn, "lexeme", morph.lemma, stability=0.15)
        self.repository.ensure_global_placement(conn, lexeme, global_space_id)
        conn.execute(
            """INSERT INTO v2_lexemes(cloud_id, lemma, language, pos_tag, semantic_state)
            VALUES (?, ?, 'ru', ?, 'candidate') ON CONFLICT(cloud_id) DO UPDATE SET pos_tag = excluded.pos_tag""",
            (lexeme["id"], morph.lemma, morph.pos_tag),
        )
        word, word_created = self.repository.get_or_create_cloud(conn, "word_form", token.normalized, stability=0.12)
        self.repository.ensure_global_placement(conn, word, global_space_id)
        conn.execute(
            """INSERT INTO v2_word_forms(cloud_id, normalized_form, language, lexeme_cloud_id, pos_tag, morphology_json)
            VALUES (?, ?, 'ru', ?, ?, ?) ON CONFLICT(cloud_id) DO UPDATE SET
            lexeme_cloud_id = excluded.lexeme_cloud_id, pos_tag = excluded.pos_tag, morphology_json = excluded.morphology_json""",
            (word["id"], token.normalized, lexeme["id"], morph.pos_tag,
             encode({"lemma": morph.lemma, "pos": morph.pos_tag, **morph.features})),
        )
        character_clouds = []
        for character in token.characters:
            char_cloud, _ = self.repository.get_or_create_cloud(conn, "character", character.normalized, stability=0.05)
            self.repository.ensure_global_placement(conn, char_cloud, global_space_id)
            character_clouds.append(char_cloud)
        if word_created:
            self.repository.ensure_space(conn, "word_structure_space", int(word["id"]), global_space_id)
            count = len(character_clouds)
            for index, char_cloud in enumerate(character_clouds):
                x = 0.0 if count == 1 else -36.0 * (count - 1) / 2.0 + 36.0 * index
                self.repository.add_component(conn, int(word["id"]), int(char_cloud["id"]), index, "character", x, 0.0)
        return word, lexeme

    def _ensure_candidate(self, conn: Any, lexeme: Dict[str, Any], global_space_id: int) -> None:
        candidate, _ = self.repository.get_or_create_cloud(
            conn, "concept_candidate", lexeme["canonical_name"], mass=0.5, density=0.6, stability=0.05
        )
        self.repository.ensure_global_placement(conn, candidate, global_space_id)
        conn.execute(
            """INSERT INTO v2_semantic_memberships(lexeme_cloud_id, concept_cloud_id, weight, confidence, evidence_count, updated_at)
            VALUES (?, ?, 1, 0.5, 1, ?) ON CONFLICT(lexeme_cloud_id, concept_cloud_id) DO UPDATE SET
            evidence_count = evidence_count + 1, confidence = MIN(1, confidence + 0.02), updated_at = excluded.updated_at""",
            (lexeme["id"], candidate["id"], utcnow()),
        )
