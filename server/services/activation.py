"""Activation management for nebula system."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from server.models.cloud import Cloud, CloudPlacement
from server.database import get_connection


@dataclass
class ActivationManager:
    """Manages cloud activation lifecycle."""
    
    decay_rate: float = 0.95
    min_activation: float = 0.01
    spread_factor: float = 0.3
    max_spread_distance: float = 200.0
    
    # Session tracking
    _session_id: str = ""
    _sequence_counter: int = 0
    _context_window_id: str = ""
    _coactivation_buffer: Dict[Tuple[int, int], List[Tuple[int, float]]] = field(default_factory=dict)
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new activation session."""
        import uuid
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._sequence_counter = 0
        self._context_window_id = ""
        self._coactivation_buffer.clear()
        return self._session_id
    
    def set_context_window(self, context_id: str) -> None:
        """Set context window for grouping related activations."""
        self._context_window_id = context_id
    
    def activate_cloud(self, cloud: Cloud, value: float = 1.0) -> None:
        """Activate a global cloud."""
        value = max(0.0, float(value))
        cloud.activation = min(1.0, cloud.activation + value)
        cloud.last_activated_at = datetime.utcnow()
        cloud.observation_count += 1

        with get_connection() as conn:
            conn.execute(
                """UPDATE clouds SET
                activation = MIN(1.0, activation + ?),
                observation_count = observation_count + 1,
                last_activated_at = ?, updated_at = ?
                WHERE id = ?""",
                (value, cloud.last_activated_at.isoformat(),
                 cloud.last_activated_at.isoformat(), cloud.id),
            )
            conn.execute(
                """UPDATE cloud_placements SET
                activation = MIN(1.0, activation + ?),
                updated_at = ?
                WHERE cloud_id = ?""",
                (value, cloud.last_activated_at.isoformat(), cloud.id),
            )
            conn.commit()
        
        # Record event
        self._sequence_counter += 1
        record_activation_event(
            self._session_id, cloud.id, None, cloud.layer_id, value,
            self._sequence_counter, self._context_window_id
        )
        
        # Track co-activation
        self._track_coactivation(cloud.id, value)
    
    def activate_placement(self, placement: CloudPlacement, value: float = 1.0) -> None:
        """Activate a local cloud placement."""
        placement.activation = min(1.0, placement.activation + value)
        # Also activate the global cloud
        # (in practice, would sync with cloud_repo)
    
    def _track_coactivation(self, cloud_id: int, value: float) -> None:
        """Track co-activation with other recently active clouds."""
        # Find other clouds activated in the same context window
        from server.database import get_connection
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT cloud_id, activation_value, sequence_position, layer_id FROM activation_events
                WHERE session_id = ? AND context_window_id = ? AND cloud_id != ?
                ORDER BY sequence_position DESC LIMIT 10""",
                (self._session_id, self._context_window_id, cloud_id)
            ).fetchall()
            
            for row in rows:
                other_id = row["cloud_id"]
                other_value = row["activation_value"]
                seq_dist = abs(self._sequence_counter - row["sequence_position"])
                layer_id = row["layer_id"]
                
                # Update co-activation stats
                update_coactivation_stats(
                    cloud_id, other_id, layer_id,
                    sequence_distance=seq_dist,
                    weight=value * other_value
                )
    
    def decay_activations(self, placements: List[CloudPlacement]) -> None:
        """Decay activation for a list of placements."""
        for p in placements:
            p.activation *= self.decay_rate
            if p.activation < self.min_activation:
                p.activation = 0.0
    
    def get_recently_active(self, layer_id: int, hours: float = 1.0, limit: int = 100) -> List[Cloud]:
        """Get recently activated clouds in a layer."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM clouds 
                WHERE layer_id = ? AND last_activated_at >= ?
                ORDER BY last_activated_at DESC LIMIT ?""",
                (layer_id, cutoff.isoformat(), limit)
            ).fetchall()
        
        from server.repositories.cloud_repository import CloudRepository
        repo = CloudRepository()
        return [repo._row_to_cloud(row) for row in rows]


def spread_activation_in_space(grid, placements: Dict, source_id: int, 
                               source_activation: float, config) -> None:
    """
    Spread activation from a source placement to nearby placements.
    This creates the "glow" effect around active concepts.
    """
    if source_activation <= 0:
        return
    
    # Find nearby placements
    source = placements.get(source_id)
    if not source:
        return
    
    nearby = grid.get_nearby(source_id, config.max_spread_distance)
    
    for target_id, distance in nearby:
        if target_id == source_id:
            continue
        
        target = placements.get(target_id)
        if not target:
            continue
        
        # Activation spread decreases with distance
        spread = source_activation * config.activation_spread_factor
        spread *= max(0.0, 1.0 - distance / config.max_spread_distance)
        
        # Also influenced by overlap
        from server.services.spatial_index import compute_overlap
        overlap = compute_overlap(source, target)
        spread *= (1.0 + overlap)
        
        # Apply
        target.activation = min(1.0, target.activation + spread)


def record_activation_event(session_id: str, cloud_id: int, placement_id: Optional[int],
                            layer_id: int, activation_value: float,
                            sequence_position: int, context_window_id: str = None,
                            metadata: Dict = None) -> None:
    """Record an activation event for learning."""
    import json
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO activation_events
            (session_id, cloud_id, placement_id, layer_id, activation_value,
             sequence_position, timestamp, context_window_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, cloud_id, placement_id, layer_id, activation_value,
             sequence_position, datetime.utcnow().isoformat(), context_window_id,
             json.dumps(metadata or {}))
        )
        conn.commit()


def update_coactivation_stats(cloud_a_id: int, cloud_b_id: int, layer_id: int,
                              sequence_distance: int = 0, weight: float = 1.0) -> None:
    """Update co-activation statistics for two clouds."""
    import json
    with get_connection() as conn:
        # Ensure consistent ordering
        a_id, b_id = sorted([cloud_a_id, cloud_b_id])
        
        row = conn.execute(
            """SELECT * FROM coactivation_stats 
            WHERE cloud_a_id = ? AND cloud_b_id = ? AND layer_id = ?""",
            (a_id, b_id, layer_id)
        ).fetchone()
        
        if row:
            new_count = row["coactivation_count"] + 1
            # Weighted score decays slowly
            new_score = row["weighted_score"] * 0.99 + weight * 0.01
            # Running average of sequence distance
            total_dist = row["average_sequence_distance"] * row["coactivation_count"]
            new_avg_dist = (total_dist + sequence_distance) / new_count
            
            conn.execute(
                """UPDATE coactivation_stats SET
                coactivation_count = ?, weighted_score = ?, average_sequence_distance = ?,
                last_updated_at = ?
                WHERE cloud_a_id = ? AND cloud_b_id = ? AND layer_id = ?""",
                (new_count, new_score, new_avg_dist, datetime.utcnow().isoformat(),
                 a_id, b_id, layer_id)
            )
        else:
            conn.execute(
                """INSERT INTO coactivation_stats
                (cloud_a_id, cloud_b_id, layer_id, coactivation_count, weighted_score,
                 average_sequence_distance, last_updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?)""",
                (a_id, b_id, layer_id, weight, float(sequence_distance),
                 datetime.utcnow().isoformat())
            )
        conn.commit()


def get_coactivation_neighbors(cloud_id: int, layer_id: int, 
                                min_score: float = 0.1, limit: int = 20) -> List[Tuple[int, float]]:
    """Get clouds with high co-activation scores."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT cloud_a_id, cloud_b_id, weighted_score 
            FROM coactivation_stats
            WHERE (cloud_a_id = ? OR cloud_b_id = ?) AND layer_id = ? AND weighted_score >= ?
            ORDER BY weighted_score DESC LIMIT ?""",
            (cloud_id, cloud_id, layer_id, min_score, limit)
        ).fetchall()
    
    result = []
    for row in rows:
        other_id = row["cloud_b_id"] if row["cloud_a_id"] == cloud_id else row["cloud_a_id"]
        result.append((other_id, row["weighted_score"]))
    return result


def compute_activation_from_text(text: str, layer_name: str = "word_form") -> Dict[int, float]:
    """
    Compute activation values for clouds based on text input.
    Returns cloud_id -> activation_value.
    """
    from server.tokenizer import tokenize_hierarchical
    from server.repositories.cloud_repository import CloudRepository, LayerRepository
    
    result = {}
    tokenization = tokenize_hierarchical(text)
    
    cloud_repo = CloudRepository()
    layer_repo = LayerRepository()
    layer = layer_repo.get_by_name(layer_name)
    
    if not layer:
        return result
    
    if layer_name == "character":
        # Activate character clouds
        for char_token in tokenization.all_characters:
            cloud = cloud_repo.get_by_canonical_name(layer.id, char_token.normalized)
            if cloud:
                result[cloud.id] = 1.0
    elif layer_name == "word_form":
        # Activate word form clouds
        for word_token in tokenization.all_tokens:
            cloud = cloud_repo.get_by_canonical_name(layer.id, word_token.normalized)
            if cloud:
                # Higher activation for first occurrence
                result[cloud.id] = result.get(cloud.id, 0) + 1.0
    
    # Normalize
    if result:
        max_act = max(result.values())
        for k in result:
            result[k] = result[k] / max_act
    
    return result
