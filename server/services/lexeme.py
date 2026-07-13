"""Lexeme service - handles Russian normalization, context vectors, and concept formation."""

import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pymorphy3 import MorphAnalyzer

from server.database import get_connection, now
from server.models.cloud import Cloud
from server.repositories.cloud_repository import CloudRepository, LayerRepository


# Initialize pymorphy3 for Russian
morph = MorphAnalyzer(lang='ru')


@dataclass
class Lexeme:
    """Normalized lexeme with morphological info."""
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
    def features(self, value: Dict[str, Any]):
        self.features_json = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    
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
    """Service for managing lexemes, context vectors, and concept formation."""
    
    def __init__(self):
        self.cloud_repo = CloudRepository()
        self.layer_repo = LayerRepository()
    
    def get_layer_id(self, layer_name: str) -> Optional[int]:
        layer = self.layer_repo.get_by_name(layer_name)
        return layer.id if layer else None
    
    # ============================================================
    # Russian Normalization with pymorphy3
    # ============================================================
    
    def normalize_russian(self, word: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Normalize Russian word using pymorphy3.
        Returns (canonical_form, pos_tag, features_dict).
        """
        parsed = morph.parse(word)
        if not parsed:
            return word.casefold(), None, {}
        
        # Get the most probable parse
        best = parsed[0]
        canonical = best.normal_form
        pos = best.tag.POS
        features = {
            "tag": str(best.tag),
            "score": float(best.score),
            "methods": [str(m) for m in best.methods_stack],
        }
        return canonical, pos, features
    
    def normalize_unknown(self, word: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Normalize unknown/Latin words using casefold."""
        return word.casefold(), None, {"method": "casefold"}
    
    def get_or_create_lexeme(self, word_form: str, language: str = "ru") -> Lexeme:
        """Get or create lexeme for a word form."""
        if language == "ru":
            canonical, pos, features = self.normalize_russian(word_form)
        else:
            canonical, pos, features = self.normalize_unknown(word_form)
        
        with get_connection() as conn:
            # Try to find existing lexeme
            row = conn.execute(
                "SELECT * FROM lexemes WHERE canonical_form = ? AND language = ?",
                (canonical, language)
            ).fetchone()
            
            if row:
                lexeme = self._row_to_lexeme(row)
                # Update frequency
                conn.execute(
                    "UPDATE lexemes SET frequency = frequency + 1, updated_at = ? WHERE id = ?",
                    (now(), lexeme.id)
                )
                conn.commit()
                return lexeme
            
            # Create new lexeme
            cursor = conn.execute(
                """INSERT INTO lexemes
                (canonical_form, language, pos_tag, features_json, frequency, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (canonical, language, pos, json.dumps(features, separators=(",", ":"), ensure_ascii=False),
                 now(), now())
            )
            lexeme_id = cursor.lastrowid
            conn.commit()
            
            return Lexeme(
                id=lexeme_id,
                canonical_form=canonical,
                language=language,
                pos_tag=pos,
                features_json=json.dumps(features, separators=(",", ":"), ensure_ascii=False),
                frequency=1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
    
    def link_word_form_to_lexeme(self, word_form_cloud_id: int, lexeme_id: int, is_canonical: bool = False) -> None:
        """Link a word_form cloud to its lexeme."""
        with get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO word_form_to_lexeme
                (word_form_cloud_id, lexeme_id, is_canonical, created_at)
                VALUES (?, ?, ?, ?)""",
                (word_form_cloud_id, lexeme_id, 1 if is_canonical else 0, now())
            )
            conn.commit()
    
    def get_lexeme_for_word_form(self, word_form_cloud_id: int) -> Optional[Lexeme]:
        """Get the lexeme linked to a word form cloud."""
        with get_connection() as conn:
            row = conn.execute(
                """SELECT l.* FROM lexemes l
                JOIN word_form_to_lexeme wfl ON l.id = wfl.lexeme_id
                WHERE wfl.word_form_cloud_id = ?
                ORDER BY wfl.is_canonical DESC LIMIT 1""",
                (word_form_cloud_id,)
            ).fetchone()
        return self._row_to_lexeme(row) if row else None
    
    def get_word_forms_for_lexeme(self, lexeme_id: int) -> List[int]:
        """Get all word form cloud IDs for a lexeme."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT word_form_cloud_id FROM word_form_to_lexeme WHERE lexeme_id = ?",
                (lexeme_id,)
            ).fetchall()
        return [row["word_form_cloud_id"] for row in rows]
    
    # ============================================================
    # Context Vector Accumulation with PPMI
    # ============================================================
    
    def accumulate_context(self, sentence_lexeme_ids: List[int], window_size: int = 5) -> None:
        """
        Accumulate context vectors for lexemes in a sentence.
        Weight = 1 / (1 + distance) for neighbors within window.
        """
        if len(sentence_lexeme_ids) < 2:
            return
        
        # Count co-occurrences
        cooccur_counts: Dict[Tuple[int, int], float] = Counter()
        lexeme_totals: Counter = Counter()
        
        for i, lexeme_id in enumerate(sentence_lexeme_ids):
            lexeme_totals[lexeme_id] += 1
            # Look at neighbors within window
            for j in range(max(0, i - window_size), min(len(sentence_lexeme_ids), i + window_size + 1)):
                if i == j:
                    continue
                context_id = sentence_lexeme_ids[j]
                distance = abs(i - j)
                weight = 1.0 / (1.0 + distance)
                pair = tuple(sorted([lexeme_id, context_id]))
                cooccur_counts[pair] += weight
                lexeme_totals[context_id] += 1
        
        # Compute PPMI and update context_vectors
        total_windows = sum(lexeme_totals.values())
        
        with get_connection() as conn:
            for (lex_a, lex_b), cooccur in cooccur_counts.items():
                # PMI = log(P(a,b) / (P(a) * P(b)))
                p_ab = cooccur / total_windows
                p_a = lexeme_totals[lex_a] / total_windows
                p_b = lexeme_totals[lex_b] / total_windows
                
                if p_a > 0 and p_b > 0:
                    pmi = math.log(p_ab / (p_a * p_b))
                    ppmi = max(0.0, pmi)
                    
                    if ppmi > 0:
                        # Update both directions
                        for l1, l2 in [(lex_a, lex_b), (lex_b, lex_a)]:
                            row = conn.execute(
                                "SELECT weight, count FROM context_vectors WHERE lexeme_id = ? AND context_lexeme_id = ?",
                                (l1, l2)
                            ).fetchone()
                            
                            if row:
                                new_count = row["count"] + 1
                                # Exponential moving average for weight
                                new_weight = row["weight"] * 0.9 + ppmi * 0.1
                                conn.execute(
                                    """UPDATE context_vectors SET
                                    weight = ?, count = ?, updated_at = ?
                                    WHERE lexeme_id = ? AND context_lexeme_id = ?""",
                                    (new_weight, new_count, now(), l1, l2)
                                )
                            else:
                                conn.execute(
                                    """INSERT INTO context_vectors
                                    (lexeme_id, context_lexeme_id, weight, count, updated_at)
                                    VALUES (?, ?, ?, 1, ?)""",
                                    (l1, l2, ppmi, now())
                                )
            conn.commit()
    
    def get_context_vector(self, lexeme_id: int, top_k: int = 100) -> Dict[int, float]:
        """Get context vector for a lexeme as dict of context_lexeme_id -> weight."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT context_lexeme_id, weight FROM context_vectors
                WHERE lexeme_id = ? ORDER BY weight DESC LIMIT ?""",
                (lexeme_id, top_k)
            ).fetchall()
        return {row["context_lexeme_id"]: row["weight"] for row in rows}
    
    def cosine_similarity(self, vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0
        
        # Get common keys
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0
        
        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    # ============================================================
    # Concept Formation with Centroid Clustering
    # ============================================================
    
    def find_or_create_concept(self, lexeme_ids: List[int], min_contexts: int = 3, 
                                similarity_threshold: float = 0.72,
                                merge_threshold: float = 0.85,
                                unify_threshold: float = 0.92) -> Optional[int]:
        """
        Find existing concept or create new one from lexemes with similar context vectors.
        Returns concept_cloud_id.
        """
        if len(lexeme_ids) < 2:
            return None
        
        # Get context vectors for all lexemes
        vectors = {}
        for lid in lexeme_ids:
            vec = self.get_context_vector(lid)
            if vec:
                vectors[lid] = vec
        
        if len(vectors) < 2:
            return None
        
        # Check if we have enough distinct contexts (at least 3 different context windows)
        # For now, we check if lexemes have been seen in enough sentences
        with get_connection() as conn:
            # Check existing concepts in concept layer
            concept_layer_id = self.get_layer_id("concept")
            if not concept_layer_id:
                return None
            
            concept_clouds = conn.execute(
                "SELECT id FROM clouds WHERE layer_id = ?", (concept_layer_id,)
            ).fetchall()
            
            # Compare with existing concept centroids
            best_match = None
            best_similarity = 0.0
            
            for c_row in concept_clouds:
                centroid_row = conn.execute(
                    "SELECT centroid_vector_json, member_lexeme_ids_json FROM concept_centroids WHERE concept_cloud_id = ?",
                    (c_row["id"],)
                ).fetchone()
                
                if not centroid_row:
                    continue
                
                centroid_vec = json.loads(centroid_row["centroid_vector_json"])
                member_ids = json.loads(centroid_row["member_lexeme_ids_json"])
                
                # Compute average similarity to centroid members
                similarities = []
                for mid in member_ids:
                    if mid in vectors:
                        sim = self.cosine_similarity(vectors[mid], centroid_vec)
                        similarities.append(sim)
                
                if similarities:
                    avg_sim = sum(similarities) / len(similarities)
                    if avg_sim > best_similarity:
                        best_similarity = avg_sim
                        best_match = c_row["id"]
            
            # If good match found, add to existing concept
            if best_match and best_similarity >= similarity_threshold:
                self._add_lexemes_to_concept(best_match, list(vectors.keys()))
                return best_match
            
            # Check if we should merge with existing concept (similarity >= 0.85)
            if best_match and best_similarity >= merge_threshold:
                self._add_lexemes_to_concept(best_match, list(vectors.keys()))
                return best_match
            
            # Create new concept
            return self._create_concept_from_lexemes(list(vectors.keys()))
    
    def _create_concept_from_lexemes(self, lexeme_ids: List[int]) -> Optional[int]:
        """Create a new concept cloud from lexemes."""
        if len(lexeme_ids) < 2:
            return None
        
        # Compute centroid vector
        vectors = [self.get_context_vector(lid) for lid in lexeme_ids]
        vectors = [v for v in vectors if v]
        
        if len(vectors) < 2:
            return None
        
        # Average the vectors
        all_keys = set()
        for v in vectors:
            all_keys.update(v.keys())
        
        centroid = {}
        for key in all_keys:
            centroid[key] = sum(v.get(key, 0) for v in vectors) / len(vectors)
        
        # Generate concept name from top lexemes
        lexeme_names = []
        with get_connection() as conn:
            for lid in lexeme_ids[:5]:
                row = conn.execute("SELECT canonical_form FROM lexemes WHERE id = ?", (lid,)).fetchone()
                if row:
                    lexeme_names.append(row["canonical_form"])
        
        concept_name = " · ".join(lexeme_names[:3]) if lexeme_names else f"concept_{lexeme_ids[0]}"
        
        concept_layer_id = self.get_layer_id("concept")
        if not concept_layer_id:
            return None
        
        with get_connection() as conn:
            # Create concept cloud
            cursor = conn.execute(
                """INSERT INTO clouds
                (layer_id, cloud_type, canonical_name, mass, density, radius, stability, activation,
                 observation_count, created_at, updated_at, metadata_json)
                VALUES (?, 'concept', ?, 3.0, 1.0, 30.0, 0.3, 0.0, 1, ?, ?, '{}')""",
                (concept_layer_id, concept_name, now(), now())
            )
            concept_cloud_id = cursor.lastrowid
            
            # Store centroid
            conn.execute(
                """INSERT INTO concept_centroids
                (concept_cloud_id, centroid_vector_json, member_lexeme_ids_json, stability, created_at, updated_at)
                VALUES (?, ?, ?, 0.5, ?, ?)""",
                (concept_cloud_id, json.dumps(centroid, separators=(",", ":")),
                 json.dumps(lexeme_ids, separators=(",", ":")), now(), now())
            )
            
            # Create fuzzy memberships
            for lid in lexeme_ids:
                vec = self.get_context_vector(lid)
                if vec:
                    sim = self.cosine_similarity(vec, centroid)
                    # smoothstep(0.55, 0.85, cosine)
                    membership = self._smoothstep(0.55, 0.85, sim)
                    if membership > 0:
                        centrality = sim  # simplified
                        context_coverage = len(vec) / max(1, len(centroid))
                        conn.execute(
                            """INSERT INTO lexeme_concept_membership
                            (lexeme_id, concept_cloud_id, membership, centrality, context_coverage, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (lid, concept_cloud_id, membership, centrality, context_coverage, now())
                        )
            
            conn.commit()
        
        return concept_cloud_id
    
    def _add_lexemes_to_concept(self, concept_cloud_id: int, lexeme_ids: List[int]) -> None:
        """Add lexemes to an existing concept."""
        with get_connection() as conn:
            # Get current centroid
            centroid_row = conn.execute(
                "SELECT centroid_vector_json, member_lexeme_ids_json FROM concept_centroids WHERE concept_cloud_id = ?",
                (concept_cloud_id,)
            ).fetchone()
            
            if not centroid_row:
                return
            
            old_centroid = json.loads(centroid_row["centroid_vector_json"])
            old_members = json.loads(centroid_row["member_lexeme_ids_json"])
            
            # Add new members
            all_members = list(set(old_members + lexeme_ids))
            
            # Recompute centroid
            vectors = [self.get_context_vector(lid) for lid in all_members]
            vectors = [v for v in vectors if v]
            
            if len(vectors) < 2:
                return
            
            all_keys = set()
            for v in vectors:
                all_keys.update(v.keys())
            
            new_centroid = {}
            for key in all_keys:
                new_centroid[key] = sum(v.get(key, 0) for v in vectors) / len(vectors)
            
            # Update centroid
            conn.execute(
                """UPDATE concept_centroids SET
                centroid_vector_json = ?, member_lexeme_ids_json = ?, updated_at = ?
                WHERE concept_cloud_id = ?""",
                (json.dumps(new_centroid, separators=(",", ":")),
                 json.dumps(all_members, separators=(",", ":")),
                 now(), concept_cloud_id)
            )
            
            # Update memberships
            for lid in lexeme_ids:
                vec = self.get_context_vector(lid)
                if vec:
                    sim = self.cosine_similarity(vec, new_centroid)
                    membership = self._smoothstep(0.55, 0.85, sim)
                    if membership > 0:
                        centrality = sim
                        context_coverage = len(vec) / max(1, len(new_centroid))
                        conn.execute(
                            """INSERT OR REPLACE INTO lexeme_concept_membership
                            (lexeme_id, concept_cloud_id, membership, centrality, context_coverage, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (lid, concept_cloud_id, membership, centrality, context_coverage, now())
                        )
            
            # Update concept cloud mass
            conn.execute(
                "UPDATE clouds SET mass = mass + 1.0, observation_count = observation_count + 1, updated_at = ? WHERE id = ?",
                (now(), concept_cloud_id)
            )
            
            conn.commit()
    
    def _smoothstep(self, edge0: float, edge1: float, x: float) -> float:
        """Smoothstep interpolation."""
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)
    
    def get_concept_members(self, concept_cloud_id: int) -> List[Tuple[int, float]]:
        """Get lexeme members of a concept with their membership values."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT lexeme_id, membership, centrality, context_coverage 
                FROM lexeme_concept_membership 
                WHERE concept_cloud_id = ? AND membership > 0
                ORDER BY membership DESC""",
                (concept_cloud_id,)
            ).fetchall()
        return [(row["lexeme_id"], row["membership"]) for row in rows]
    
    def merge_concepts(self, concept_a_id: int, concept_b_id: int, threshold: float = 0.92) -> Optional[int]:
        """Merge two concepts if their centroids are similar enough."""
        with get_connection() as conn:
            centroid_a = conn.execute(
                "SELECT centroid_vector_json FROM concept_centroids WHERE concept_cloud_id = ?",
                (concept_a_id,)
            ).fetchone()
            centroid_b = conn.execute(
                "SELECT centroid_vector_json FROM concept_centroids WHERE concept_cloud_id = ?",
                (concept_b_id,)
            ).fetchone()
            
            if not centroid_a or not centroid_b:
                return None
            
            vec_a = json.loads(centroid_a["centroid_vector_json"])
            vec_b = json.loads(centroid_b["centroid_vector_json"])
            
            sim = self.cosine_similarity(vec_a, vec_b)
            
            if sim >= threshold:
                # Merge B into A
                members_a = conn.execute(
                    "SELECT lexeme_id FROM lexeme_concept_membership WHERE concept_cloud_id = ?",
                    (concept_a_id,)
                ).fetchall()
                members_b = conn.execute(
                    "SELECT lexeme_id FROM lexeme_concept_membership WHERE concept_cloud_id = ?",
                    (concept_b_id,)
                ).fetchall()
                
                all_member_ids = list(set([r["lexeme_id"] for r in members_a] + [r["lexeme_id"] for r in members_b]))
                
                # Recompute centroid
                vectors = [self.get_context_vector(lid) for lid in all_member_ids]
                vectors = [v for v in vectors if v]
                
                if len(vectors) >= 2:
                    all_keys = set()
                    for v in vectors:
                        all_keys.update(v.keys())
                    
                    new_centroid = {}
                    for key in all_keys:
                        new_centroid[key] = sum(v.get(key, 0) for v in vectors) / len(vectors)
                    
                    # Update concept A
                    conn.execute(
                        """UPDATE concept_centroids SET
                        centroid_vector_json = ?, member_lexeme_ids_json = ?, updated_at = ?
                        WHERE concept_cloud_id = ?""",
                        (json.dumps(new_centroid, separators=(",", ":")),
                         json.dumps(all_member_ids, separators=(",", ":")),
                         now(), concept_a_id)
                    )
                    
                    # Move memberships from B to A
                    for row in members_b:
                        lid = row["lexeme_id"]
                        vec = self.get_context_vector(lid)
                        if vec:
                            membership = self._smoothstep(0.55, 0.85, self.cosine_similarity(vec, new_centroid))
                            if membership > 0:
                                conn.execute(
                                    """INSERT OR REPLACE INTO lexeme_concept_membership
                                    (lexeme_id, concept_cloud_id, membership, centrality, context_coverage, updated_at)
                                    VALUES (?, ?, ?, ?, ?, ?)""",
                                    (lid, concept_a_id, membership, 
                                     self.cosine_similarity(vec, new_centroid),
                                     len(vec) / max(1, len(new_centroid)), now())
                                )
                    
                    # Delete concept B
                    conn.execute("DELETE FROM lexeme_concept_membership WHERE concept_cloud_id = ?", (concept_b_id,))
                    conn.execute("DELETE FROM concept_centroids WHERE concept_cloud_id = ?", (concept_b_id,))
                    conn.execute("DELETE FROM clouds WHERE id = ?", (concept_b_id,))
                    
                    conn.commit()
                    return concept_a_id
        
        return None
    
    def _row_to_lexeme(self, row: sqlite3.Row) -> Lexeme:
        return Lexeme(
            id=row["id"],
            canonical_form=row["canonical_form"],
            language=row["language"],
            pos_tag=row["pos_tag"],
            features_json=row["features_json"] or "{}",
            frequency=row["frequency"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )


# Global instance
lexeme_service = LexemeService()