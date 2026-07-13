"""Condensation service - creates higher-level clouds from stable lower-level patterns."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from server.database import get_connection
from server.models.cloud import Cloud, CloudPlacement, StructuralComponent
from server.repositories.cloud_repository import CloudRepository, CloudPlacementRepository, StructuralComponentRepository, LayerRepository


@dataclass
class CondensationCandidate:
    """Candidate for condensation to higher layer."""
    id: Optional[int] = None
    source_layer_id: int = 0
    target_layer_id: int = 0
    signature_hash: str = ""
    observations: int = 0
    stability: float = 0.0
    sequence_sensitive: bool = True
    proposed_cloud_id: Optional[int] = None
    status: str = "pending"  # pending, confirmed, rejected, merged
    metadata_json: str = "{}"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class CondensationService:
    """Manages condensation of lower-level clouds into higher-level clouds."""
    
    def __init__(self):
        self.cloud_repo = CloudRepository()
        self.placement_repo = CloudPlacementRepository()
        self.component_repo = StructuralComponentRepository()
        self.layer_repo = LayerRepository()
    
    def get_layer_id(self, layer_name: str) -> Optional[int]:
        layer = self.layer_repo.get_by_name(layer_name)
        return layer.id if layer else None
    
    # ============================================================
    # Character -> Word Form Condensation
    # ============================================================
    
    def compute_character_sequence_signature(self, char_cloud_ids: List[int]) -> str:
        """Compute signature for ordered character sequence."""
        sig_data = f"char_seq:{':'.join(str(cid) for cid in char_cloud_ids)}"
        return hashlib.sha256(sig_data.encode()).hexdigest()[:32]
    
    def create_word_form_from_characters(self, char_cloud_ids: List[int], 
                                          word_text: str, 
                                          min_observations: int = 2) -> Optional[Tuple[Cloud, bool]]:
        """
        Create or strengthen word_form cloud from character sequence.
        char_cloud_ids must be in correct order (position_index).
        Returns (cloud, is_new) where is_new indicates if cloud was newly created.
        """
        if not char_cloud_ids:
            return None
        
        signature = self.compute_character_sequence_signature(char_cloud_ids)
        source_layer_id = self.get_layer_id("character")
        target_layer_id = self.get_layer_id("word_form")
        
        if not source_layer_id or not target_layer_id:
            return None
        
        with get_connection() as conn:
            # Check existing candidate
            row = conn.execute(
                """SELECT * FROM condensation_candidates 
                WHERE source_layer_id = ? AND target_layer_id = ? 
                AND signature_hash = ? AND status = 'pending'""",
                (source_layer_id, target_layer_id, signature)
            ).fetchone()
            
            if row:
                # Update existing candidate
                candidate_id = row["id"]
                new_obs = row["observations"] + 1
                new_stability = min(1.0, row["stability"] + 0.1)
                
                conn.execute(
                    """UPDATE condensation_candidates SET
                    observations = ?, stability = ?, updated_at = ?
                    WHERE id = ?""",
                    (new_obs, new_stability, datetime.utcnow().isoformat(), candidate_id)
                )
                
                if new_obs >= min_observations:
                    # Confirm condensation
                    return self._confirm_condensation(candidate_id, word_text, char_cloud_ids, conn)
                
                conn.commit()
                return None
            
            # Create new candidate
            now_iso = datetime.utcnow().isoformat()
            cursor = conn.execute(
                """INSERT INTO condensation_candidates
                (source_layer_id, target_layer_id, signature_hash, observations, 
                 stability, sequence_sensitive, status, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, 1, 0.1, 1, 'pending', ?, ?, ?)""",
                (source_layer_id, target_layer_id, signature, 
                 json.dumps({"char_ids": char_cloud_ids, "word_text": word_text}),
                 now_iso, now_iso)
            )
            candidate_id = cursor.lastrowid
            conn.commit()
            
            if min_observations <= 1:
                return self._confirm_condensation(candidate_id, word_text, char_cloud_ids, conn)
            
            return None
    
    def _confirm_condensation(self, candidate_id: int, canonical_name: str,
                              child_cloud_ids: List[int], conn) -> tuple[Cloud, bool]:
        """Confirm condensation candidate and create the cloud.
        Returns (cloud, is_new) where is_new indicates if cloud was newly created.
        """
        target_layer_id = self.get_layer_id("word_form")
        
        # Check if cloud already exists
        existing = conn.execute(
            "SELECT id FROM clouds WHERE layer_id = ? AND canonical_name = ?",
            (target_layer_id, canonical_name)
        ).fetchone()
        
        is_new = False
        if existing:
            cloud_id = existing["id"]
            # Update mass and stability
            conn.execute(
                "UPDATE clouds SET mass = mass + 1.0, stability = MIN(1.0, stability + 0.1), observation_count = observation_count + 1, updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), cloud_id)
            )
        else:
            # Create new cloud
            cursor = conn.execute(
                """INSERT INTO clouds
                (layer_id, cloud_type, canonical_name, mass, density, radius, stability, activation,
                 observation_count, created_at, updated_at, metadata_json)
                VALUES (?, 'word', ?, 2.0, 1.0, 20.0, 0.5, 0.0, 1, ?, ?, '{}')""",
                (target_layer_id, canonical_name, datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
            )
            cloud_id = cursor.lastrowid
            is_new = True
        
        # Create structural components
        for idx, child_id in enumerate(child_cloud_ids):
            # Create placement in word_form's structural space
            # (We'll create the space later when zooming in)
            conn.execute(
                """INSERT INTO structural_components
                (parent_cloud_id, child_cloud_id, position_index, phase, weight, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cloud_id, child_id, idx, float(idx) * 0.1, 1.0, "character", datetime.utcnow().isoformat())
            )
        
        # Update candidate status
        conn.execute(
            """UPDATE condensation_candidates SET 
            status = 'confirmed', proposed_cloud_id = ?, updated_at = ?
            WHERE id = ?""",
            (cloud_id, datetime.utcnow().isoformat(), candidate_id)
        )
        
        conn.commit()
        
        return self.cloud_repo.get_by_id(cloud_id), is_new
    
    # ============================================================
    # Word Form -> Concept Condensation (semantic)
    # ============================================================
    
    def compute_semantic_signature(self, word_form_cloud_ids: List[int], 
                                   context_window: str = "") -> str:
        """Compute signature for semantic co-occurrence pattern."""
        # Order-insensitive for semantic co-occurrence. Runtime session IDs are
        # deliberately excluded so repeated observations can condense.
        ordered = sorted(set(word_form_cloud_ids))
        sig_data = f"semantic:{':'.join(str(cid) for cid in ordered)}"
        return hashlib.sha256(sig_data.encode()).hexdigest()[:32]
    
    def create_concept_from_word_forms(self, word_form_cloud_ids: List[int],
                                        concept_name: str,
                                        context_window: str = "",
                                        min_observations: int = 3) -> Optional[Cloud]:
        """Create concept cloud from frequently co-occurring word forms."""
        if not word_form_cloud_ids:
            return None
        
        signature = self.compute_semantic_signature(word_form_cloud_ids, context_window)
        source_layer_id = self.get_layer_id("word_form")
        target_layer_id = self.get_layer_id("concept")
        
        if not source_layer_id or not target_layer_id:
            return None
        
        with get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM condensation_candidates 
                WHERE source_layer_id = ? AND target_layer_id = ? 
                AND signature_hash = ? AND status = 'pending'""",
                (source_layer_id, target_layer_id, signature)
            ).fetchone()
            
            if row:
                candidate_id = row["id"]
                new_obs = row["observations"] + 1
                new_stability = min(1.0, row["stability"] + 0.05)
                
                conn.execute(
                    """UPDATE condensation_candidates SET
                    observations = ?, stability = ?, updated_at = ?
                    WHERE id = ?""",
                    (new_obs, new_stability, datetime.utcnow().isoformat(), candidate_id)
                )
                
                if new_obs >= min_observations:
                    return self._confirm_concept_condensation(candidate_id, concept_name, word_form_cloud_ids, conn)
                
                conn.commit()
                return None
            
            # New candidate
            cursor = conn.execute(
                """INSERT INTO condensation_candidates
                (source_layer_id, target_layer_id, signature_hash, observations, 
                 stability, sequence_sensitive, status, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, 1, 0.05, 0, 'pending', ?, ?, ?)""",
                (source_layer_id, target_layer_id, signature,
                 json.dumps({"word_form_ids": word_form_cloud_ids, "concept_name": concept_name, "context": context_window}),
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
            )
            candidate_id = cursor.lastrowid
            conn.commit()
            
            if min_observations <= 1:
                return self._confirm_concept_condensation(candidate_id, concept_name, word_form_cloud_ids, conn)
            
            return None
    
    def _confirm_concept_condensation(self, candidate_id: int, canonical_name: str,
                                       child_cloud_ids: List[int], conn) -> Cloud:
        """Confirm concept condensation."""
        target_layer_id = self.get_layer_id("concept")
        
        existing = conn.execute(
            "SELECT id FROM clouds WHERE layer_id = ? AND canonical_name = ?",
            (target_layer_id, canonical_name)
        ).fetchone()
        
        if existing:
            cloud_id = existing["id"]
            conn.execute(
                "UPDATE clouds SET mass = mass + 1.0, stability = MIN(1.0, stability + 0.05), observation_count = observation_count + 1, updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), cloud_id)
            )
        else:
            cursor = conn.execute(
                """INSERT INTO clouds
                (layer_id, cloud_type, canonical_name, mass, density, radius, stability, activation,
                 observation_count, created_at, updated_at, metadata_json)
                VALUES (?, 'concept', ?, 3.0, 1.0, 30.0, 0.3, 0.0, 1, ?, ?, '{}')""",
                (target_layer_id, canonical_name, datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
            )
            cloud_id = cursor.lastrowid
        
        # For concepts, structural components link to word forms (for structural zoom)
        for child_id in child_cloud_ids:
            conn.execute(
                """INSERT INTO structural_components
                (parent_cloud_id, child_cloud_id, position_index, phase, weight, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cloud_id, child_id, 0, 0.0, 1.0, "form", datetime.utcnow().isoformat())
            )
        
        conn.execute(
            """UPDATE condensation_candidates SET 
            status = 'confirmed', proposed_cloud_id = ?, updated_at = ?
            WHERE id = ?""",
            (cloud_id, datetime.utcnow().isoformat(), candidate_id)
        )
        
        conn.commit()
        
        return self.cloud_repo.get_by_id(cloud_id)
    
    # ============================================================
    # Co-activation -> Position Update (semantic proximity)
    # ============================================================
    
    def update_positions_from_coactivation(self, layer_name: str) -> int:
        """
        Update cloud positions based on co-activation statistics.
        This creates semantic proximity WITHOUT creating edges.
        """
        layer = self.layer_repo.get_by_name(layer_name)
        if not layer:
            return 0
        
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM coactivation_stats 
                WHERE layer_id = ? AND weighted_score > 0.1
                ORDER BY weighted_score DESC""",
                (layer.id,)
            ).fetchall()
            
            updates = 0
            for row in rows:
                cloud_a_id = row["cloud_a_id"]
                cloud_b_id = row["cloud_b_id"]
                score = row["weighted_score"]
                seq_dist = row["average_sequence_distance"]
                
                # Get placements in semantic spaces
                placements_a = conn.execute(
                    """SELECT cp.* FROM cloud_placements cp
                    JOIN spaces s ON cp.space_id = s.id
                    WHERE cp.cloud_id = ? AND s.mode = 'semantic'""",
                    (cloud_a_id,)
                ).fetchall()
                
                placements_b = conn.execute(
                    """SELECT cp.* FROM cloud_placements cp
                    JOIN spaces s ON cp.space_id = s.id
                    WHERE cp.cloud_id = ? AND s.mode = 'semantic'""",
                    (cloud_b_id,)
                ).fetchall()
                
                # Move them closer in shared spaces
                for pa in placements_a:
                    for pb in placements_b:
                        if pa.space_id == pb.space_id:
                            # They share a semantic space - move closer
                            self._move_placements_closer(pa, pb, score, conn)
                            updates += 1
            
            conn.commit()
            return updates
    
    def _move_placements_closer(self, pa: sqlite3.Row, pb: sqlite3.Row, 
                                 score: float, conn) -> None:
        """Move two placements closer based on co-activation strength."""
        import math
        
        dx = pb["x"] - pa["x"]
        dy = pb["y"] - pa["y"]
        distance = math.hypot(dx, dy)
        
        if distance < 1.0:
            return
        
        # Attraction force based on co-activation score
        force = score * 10.0 / (distance + 1.0)
        force = min(force, 5.0)  # max force
        
        move_x = force * dx / distance
        move_y = force * dy / distance
        
        # Update both towards each other
        new_ax = pa["x"] + move_x * 0.5
        new_ay = pa["y"] + move_y * 0.5
        new_bx = pb["x"] - move_x * 0.5
        new_by = pb["y"] - move_y * 0.5
        
        conn.execute(
            "UPDATE cloud_placements SET x = ?, y = ?, updated_at = ? WHERE id = ?",
            (new_ax, new_ay, datetime.utcnow().isoformat(), pa["id"])
        )
        conn.execute(
            "UPDATE cloud_placements SET x = ?, y = ?, updated_at = ? WHERE id = ?",
            (new_bx, new_by, datetime.utcnow().isoformat(), pb["id"])
        )


# Global instance
condensation_service = CondensationService()
