"""Multi-layer training manager for recursive nebula system."""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from server.database import init_db, get_stats, reset_space, get_connection, now
from server.tokenizer import tokenize_hierarchical, TokenizationResult, WordToken, CharacterToken
import json
from server.repositories.cloud_repository import (
    CloudRepository, CloudPlacementRepository, StructuralComponentRepository,
    LayerRepository, SpaceRepository,
)
from server.models.cloud import Cloud, CloudPlacement, StructuralComponent
from server.models.space import Space, Layer
from server.services.condensation import condensation_service
from server.services.activation import ActivationManager, compute_activation_from_text, record_activation_event, update_coactivation_stats
from server.services.lexeme import lexeme_service
from server.physics import LocalSpacePhysics, create_space_physics, PhysicsConfig


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    min_word_observations: int = 1
    min_concept_observations: int = 3
    min_scene_observations: int = 5
    activation_decay: float = 0.95
    coactivation_weight: float = 1.0
    physics_ticks_per_step: int = 5
    enable_character_layer: bool = True
    enable_word_form_layer: bool = True
    enable_lexeme_layer: bool = True
    enable_concept_layer: bool = True
    enable_scene_layer: bool = True


class TrainingManager:
    """Manages multi-layer learning for the nebula system."""
    
    def __init__(self, config: TrainingConfig = None):
        if config is None or isinstance(config, TrainingConfig):
            self.config = config or TrainingConfig()
        else:
            self.config = TrainingConfig(
                physics_ticks_per_step=max(1, min(5, int(getattr(config, "steps", 5))))
            )
        init_db()
        
        # Repositories
        self.cloud_repo = CloudRepository()
        self.placement_repo = CloudPlacementRepository()
        self.component_repo = StructuralComponentRepository()
        self.layer_repo = LayerRepository()
        
        # Layer IDs
        self.layer_ids = {}
        self._load_layer_ids()
        
        # Activation manager
        self.activation_manager = ActivationManager(
            decay_rate=self.config.activation_decay
        )
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.sequence_counter = 0
    
    def _load_layer_ids(self) -> None:
        """Load layer IDs from database."""
        for name in ["signal", "character", "word_form", "lexeme", "concept", "scene", "context"]:
            layer = self.layer_repo.get_by_name(name)
            if layer:
                self.layer_ids[name] = layer.id
    
    def get_layer_id(self, name: str) -> Optional[int]:
        return self.layer_ids.get(name)
    
    def _states(self) -> List:
        """Legacy compatibility."""
        return []
    
    @staticmethod
    def _serialize(states: List) -> List[Dict[str, Any]]:
        return []
    
    def learn(self, text: str) -> Dict[str, Any]:
        """
        Main learning entry point.
        Processes text through all enabled layers:
        signal -> character -> word_form -> lexeme -> concept -> scene -> context
        """
        started = time.time()
        
        # Hierarchical tokenization
        tokenization = tokenize_hierarchical(text)
        
        if not tokenization.all_tokens:
            return {
                "success": False,
                "concepts": [],
                "stats": get_stats(),
                "time_ms": 0,
                "error": "No valid tokens found"
            }
        
        # Start activation session
        self.activation_manager.start_session(self.session_id)
        
        results = {
            "created_clouds": [],
            "strengthened_clouds": [],
            "new_candidates": [],
            "position_changes": 0,
            "activations": [],
        }
        
        # Process each sentence as a context window
        for sent_idx, sentence in enumerate(tokenization.sentences):
            context_id = f"{self.session_id}:{sent_idx}"
            self.activation_manager.set_context_window(context_id)
            
            # Learn characters (if enabled)
            if self.config.enable_character_layer:
                self._learn_characters(sentence, context_id, results)
            
            # Learn word forms (if enabled)
            if self.config.enable_word_form_layer:
                self._learn_word_forms(sentence, context_id, results)
            
            # Learn lexemes and accumulate context (if enabled)
            sentence_lexeme_ids = []
            if self.config.enable_lexeme_layer:
                sentence_lexeme_ids = self._learn_lexemes(sentence, context_id, results)
            
            # Learn concepts from lexeme context vectors (if enabled)
            if self.config.enable_concept_layer and sentence_lexeme_ids:
                self._learn_concepts(sentence_lexeme_ids, context_id, results)
            
            # Create scene from word forms (if enabled)
            if self.config.enable_scene_layer:
                self._create_scene(sentence, context_id, results)
            
            # Run local physics for active spaces
            self._run_physics_for_active_spaces(results)
        
        # Update co-activation statistics
        self._update_coactivation_stats(results)
        
        # Get final space state for response
        concepts = self._get_response_concepts()
        
        return {
            "success": True,
            "concepts": concepts,
            "stats": {**get_stats(), "tokens": len(tokenization.all_tokens)},
            "time_ms": int((time.time() - started) * 1000),
            "details": results,
        }
    
    def _learn_characters(self, sentence: "SentenceTokens", context_id: str, results: Dict) -> None:
        """Learn character-level clouds."""
        layer_id = self.get_layer_id("character")
        if not layer_id:
            return
        
        for word in sentence.tokens:
            for char_token in word.characters:
                # Get or create character cloud
                cloud = self.cloud_repo.get_by_canonical_name(layer_id, char_token.normalized)
                
                if cloud:
                    # Strengthen existing
                    self.cloud_repo.increment_observation(cloud.id, mass_delta=0.05, stability_delta=0.01)
                    self.activation_manager.activate_cloud(cloud, 0.5)
                    results["strengthened_clouds"].append({
                        "id": cloud.id, "name": cloud.canonical_name, "layer": "character"
                    })
                else:
                    # Create new character cloud
                    cloud = self.cloud_repo.create(Cloud(
                        layer_id=layer_id,
                        cloud_type="character",
                        canonical_name=char_token.normalized,
                        mass=1.0,
                        density=1.0,
                        stability=0.1,
                        observation_count=1,
                    ))
                    # Create placement in global character space
                    self._ensure_global_space_and_placement(cloud, layer_id)
                    results["created_clouds"].append({
                        "id": cloud.id, "name": cloud.canonical_name, "layer": "character"
                    })
                self._ensure_global_space_and_placement(cloud, layer_id)
                
                # Record activation
                record_activation_event(
                    self.session_id, cloud.id, None, layer_id, 0.5,
                    self.sequence_counter, context_id
                )
                self.sequence_counter += 1
    
    def _learn_word_forms(self, sentence: "SentenceTokens", context_id: str, results: Dict) -> None:
        """Learn word form clouds from character sequences."""
        char_layer_id = self.get_layer_id("character")
        word_layer_id = self.get_layer_id("word_form")
        
        if not char_layer_id or not word_layer_id:
            return
        
        for word in sentence.tokens:
            # Get character clouds in order
            char_clouds = []
            for char_token in word.characters:
                cloud = self.cloud_repo.get_by_canonical_name(char_layer_id, char_token.normalized)
                if cloud:
                    char_clouds.append(cloud)
            
            if not char_clouds:
                continue
            
            # Try condensation
            result = condensation_service.create_word_form_from_characters(
                [c.id for c in char_clouds], word.normalized, 
                min_observations=self.config.min_word_observations
            )
            
            if result:
                word_cloud, is_new = result
                if is_new or word_cloud.id in [c["id"] for c in results["created_clouds"]]:
                    # Was just created
                    results["created_clouds"].append({
                        "id": word_cloud.id, "name": word_cloud.canonical_name, "layer": "word_form"
                    })
                    
                    # Create structural components linking to characters
                    self._create_structural_links(word_cloud, char_clouds)
                    
                    # Ensure word form has structural space
                    self._ensure_structural_space(word_cloud, char_layer_id)
                else:
                    # Existing word form strengthened
                    self.cloud_repo.increment_observation(word_cloud.id, mass_delta=0.1, stability_delta=0.02)
                    results["strengthened_clouds"].append({
                        "id": word_cloud.id, "name": word_cloud.canonical_name, "layer": "word_form"
                    })

                self._ensure_global_space_and_placement(word_cloud, word_layer_id)
                
                # Activate
                self.activation_manager.activate_cloud(word_cloud, 1.0)
                record_activation_event(
                    self.session_id, word_cloud.id, None, word_layer_id, 1.0,
                    self.sequence_counter, context_id
                )
                self.sequence_counter += 1
    
    def _learn_lexemes(self, sentence: "SentenceTokens", context_id: str, results: Dict) -> List[int]:
        """
        Learn lexemes from word forms and accumulate context vectors.
        Returns list of lexeme IDs for this sentence.
        """
        word_layer_id = self.get_layer_id("word_form")
        lexeme_layer_id = self.get_layer_id("lexeme")
        
        if not word_layer_id or not lexeme_layer_id:
            return []
        
        sentence_lexeme_ids = []
        
        for word in sentence.tokens:
            # Get word form cloud
            word_cloud = self.cloud_repo.get_by_canonical_name(word_layer_id, word.normalized)
            if not word_cloud:
                continue
            
            # Get or create lexeme for this word form
            lexeme = lexeme_service.get_or_create_lexeme(word.normalized)
            sentence_lexeme_ids.append(lexeme.id)
            
            # Link word form to lexeme
            lexeme_service.link_word_form_to_lexeme(word_cloud.id, lexeme.id, is_canonical=True)
            
            # Create or get lexeme cloud in lexeme layer
            lexeme_cloud = self.cloud_repo.get_by_canonical_name(lexeme_layer_id, lexeme.canonical_form)
            if not lexeme_cloud:
                lexeme_cloud = self.cloud_repo.create(Cloud(
                    layer_id=lexeme_layer_id,
                    cloud_type="lexeme",
                    canonical_name=lexeme.canonical_form,
                    mass=1.0,
                    density=1.0,
                    stability=0.2,
                    observation_count=1,
                ))
                results["created_clouds"].append({
                    "id": lexeme_cloud.id, "name": lexeme_cloud.canonical_name, "layer": "lexeme"
                })
            else:
                self.cloud_repo.increment_observation(lexeme_cloud.id, mass_delta=0.1, stability_delta=0.01)
                results["strengthened_clouds"].append({
                    "id": lexeme_cloud.id, "name": lexeme_cloud.canonical_name, "layer": "lexeme"
                })
            
            self._ensure_global_space_and_placement(lexeme_cloud, lexeme_layer_id)
            
            # Activate lexeme
            self.activation_manager.activate_cloud(lexeme_cloud, 0.9)
            record_activation_event(
                self.session_id, lexeme_cloud.id, None, lexeme_layer_id, 0.9,
                self.sequence_counter, context_id
            )
            self.sequence_counter += 1
        
        # Accumulate context vectors for this sentence
        if len(sentence_lexeme_ids) >= 2:
            lexeme_service.accumulate_context(sentence_lexeme_ids)
        
        return sentence_lexeme_ids
    
    def _learn_concepts(self, sentence_lexeme_ids: List[int], context_id: str, results: Dict) -> None:
        """Learn concept clouds from lexeme context vectors."""
        concept_layer_id = self.get_layer_id("concept")
        
        if not concept_layer_id:
            return
        
        if len(sentence_lexeme_ids) < 2:
            return
        
        # Find or create concept from lexemes with similar context vectors
        concept_cloud_id = lexeme_service.find_or_create_concept(
            sentence_lexeme_ids,
            min_contexts=3,
            similarity_threshold=0.72,
            merge_threshold=0.85,
            unify_threshold=0.92
        )
        
        if concept_cloud_id:
            concept_cloud = self.cloud_repo.get_by_id(concept_cloud_id)
            if concept_cloud:
                if concept_cloud.id not in [c["id"] for c in results["created_clouds"]]:
                    results["created_clouds"].append({
                        "id": concept_cloud.id, "name": concept_cloud.canonical_name, "layer": "concept"
                    })
                    # Ensure concept has semantic space
                    self._ensure_semantic_space(concept_cloud, concept_layer_id)
                else:
                    results["strengthened_clouds"].append({
                        "id": concept_cloud.id, "name": concept_cloud.canonical_name, "layer": "concept"
                    })
                
                # Activate concept
                self.activation_manager.activate_cloud(concept_cloud, 0.95)
                record_activation_event(
                    self.session_id, concept_cloud.id, None, concept_layer_id, 0.95,
                    self.sequence_counter, context_id
                )
                self.sequence_counter += 1
    
    def _create_scene(self, sentence: "SentenceTokens", context_id: str, results: Dict) -> None:
        """Create scene cloud from ordered word forms in a sentence."""
        word_layer_id = self.get_layer_id("word_form")
        scene_layer_id = self.get_layer_id("scene")
        
        if not word_layer_id or not scene_layer_id:
            return
        
        # Get word form clouds in order
        word_clouds = []
        word_form_ids = []
        lexeme_ids = []
        
        for word in sentence.tokens:
            cloud = self.cloud_repo.get_by_canonical_name(word_layer_id, word.normalized)
            if cloud:
                word_clouds.append(cloud)
                word_form_ids.append(cloud.id)
                # Get lexeme for this word form
                lexeme = lexeme_service.get_lexeme_for_word_form(cloud.id)
                if lexeme:
                    lexeme_ids.append(lexeme.id)
        
        if len(word_clouds) < 2:
            return
        
        # Create scene cloud
        scene_name = " ".join([c.canonical_name for c in word_clouds[:5]])
        if len(word_clouds) > 5:
            scene_name += "..."
        
        # Check if similar scene exists
        with get_connection() as conn:
            existing = conn.execute(
                """SELECT s.*, c.canonical_name FROM scenes s
                JOIN clouds c ON s.scene_cloud_id = c.id
                WHERE c.layer_id = ?""",
                (scene_layer_id,)
            ).fetchall()
            
            # Check if scene with same name already exists
            scene_cloud = None
            for row in existing:
                if row["canonical_name"] == scene_name:
                    scene_cloud = self.cloud_repo.get_by_id(row["scene_cloud_id"])
                    break
            
            if not scene_cloud:
                # Create new scene cloud
                scene_cloud = self.cloud_repo.create(Cloud(
                    layer_id=scene_layer_id,
                    cloud_type="scene",
                    canonical_name=scene_name,
                    mass=2.0,
                    density=1.0,
                    radius=40.0,
                    stability=0.4,
                    observation_count=1,
                ))
                
                # Store scene with ordered word forms and lexemes
                conn.execute(
                    """INSERT INTO scenes
                    (scene_cloud_id, sentence_text, word_form_cloud_ids_json, lexeme_ids_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (scene_cloud.id, sentence.text, 
                     json.dumps(word_form_ids, separators=(",", ":")),
                     json.dumps(lexeme_ids, separators=(",", ":")),
                     now(), now())
                )
            else:
                # Update existing scene
                conn.execute(
                    """UPDATE scenes SET
                    sentence_text = ?, word_form_cloud_ids_json = ?, lexeme_ids_json = ?, updated_at = ?
                    WHERE scene_cloud_id = ?""",
                    (sentence.text, 
                     json.dumps(word_form_ids, separators=(",", ":")),
                     json.dumps(lexeme_ids, separators=(",", ":")),
                     now(), scene_cloud.id)
                )
            conn.commit()
        
        results["created_clouds"].append({
            "id": scene_cloud.id, "name": scene_cloud.canonical_name, "layer": "scene"
        })
        
        self._ensure_global_space_and_placement(scene_cloud, scene_layer_id)
        
        # Activate scene
        self.activation_manager.activate_cloud(scene_cloud, 0.8)
        record_activation_event(
            self.session_id, scene_cloud.id, None, scene_layer_id, 0.8,
            self.sequence_counter, context_id
        )
        self.sequence_counter += 1
    
    def _create_structural_links(self, parent: "Cloud", children: List["Cloud"]) -> None:
        """Create structural components linking parent to children."""
        # Ensure parent has structural space
        structural_space = self.placement_repo.get_structural_space(parent.id)
        if not structural_space:
            # Create structural space for this word
            space = Space(
                host_cloud_id=parent.id,
                layer_id=self.get_layer_id("character"),
                mode="structural",
                coordinate_dimensions=2,
                scale=1.0,
            )
            space_repo = SpaceRepository()
            structural_space = space_repo.create(space)
        
        # Place children in structural space
        for idx, child in enumerate(children):
            # Get or create placement for child in this structural space
            placements = self.placement_repo.get_by_cloud(child.id)
            child_placement = None
            for p in placements:
                if p.space_id == structural_space.id:
                    child_placement = p
                    break
            
            if not child_placement:
                # Create new placement
                child_placement = self.placement_repo.create(CloudPlacement(
                    space_id=structural_space.id,
                    cloud_id=child.id,
                    x=400 + idx * 100,  # spaced horizontally
                    y=300,
                    radius=child.radius,
                    density=child.density,
                    mass=child.mass,
                    activation=0.0,
                ))
            
            # Create structural component
            self.component_repo.create(StructuralComponent(
                parent_cloud_id=parent.id,
                child_cloud_id=child.id,
                child_placement_id=child_placement.id,
                position_index=idx,
                phase=float(idx) * 0.1,
                weight=1.0,
                role="character",
            ))
    
    def _ensure_global_space_and_placement(self, cloud: "Cloud", layer_id: int) -> None:
        """Ensure cloud has a placement in the global layer space."""
        from server.repositories.cloud_repository import SpaceRepository

        space_repo = SpaceRepository()
        space = space_repo.get_global_space(layer_id)
        if not space:
            space = space_repo.create(Space(
                host_cloud_id=0,
                layer_id=layer_id,
                mode="global",
                coordinate_dimensions=2,
                scale=1.0,
            ))

        for placement in self.placement_repo.get_by_cloud(cloud.id):
            if placement.space_id == space.id:
                placement.activation = max(placement.activation, cloud.activation)
                placement.mass = cloud.mass
                placement.radius = cloud.radius
                placement.density = cloud.density
                self.placement_repo.update(placement)
                return

        count = len(self.placement_repo.get_by_space(space.id))
        columns = 6
        x = 160.0 + (count % columns) * 250.0
        y = 140.0 + (count // columns) * 170.0
        self.placement_repo.create(CloudPlacement(
            space_id=space.id,
            cloud_id=cloud.id,
            x=x,
            y=y,
            radius=cloud.radius,
            density=cloud.density,
            mass=cloud.mass,
            activation=cloud.activation,
        ))
    
    def _ensure_structural_space(self, cloud: "Cloud", child_layer_id: int) -> None:
        """Ensure a cloud has a structural space for its children."""
        space_repo = SpaceRepository()
        
        existing = space_repo.get_structural_space(cloud.id)
        if not existing:
            space = Space(
                host_cloud_id=cloud.id,
                layer_id=child_layer_id,
                mode="structural",
                coordinate_dimensions=2,
                scale=1.0,
            )
            space_repo.create(space)
    
    def _ensure_semantic_space(self, cloud: "Cloud", layer_id: int) -> None:
        """Ensure a cloud has a semantic space for projections."""
        space_repo = SpaceRepository()
        
        existing = space_repo.get_semantic_space(cloud.id)
        if not existing:
            space = Space(
                host_cloud_id=cloud.id,
                layer_id=layer_id,
                mode="semantic",
                coordinate_dimensions=2,
                scale=1.0,
            )
            space_repo.create(space)
    
    def _run_physics_for_active_spaces(self, results: Dict) -> None:
        """Run physics simulation for spaces with active clouds."""
        # Get all spaces that have active placements
        # For now, run physics on concept layer semantic spaces
        concept_layer_id = self.get_layer_id("concept")
        if not concept_layer_id:
            return
        
        with get_connection() as conn:
            # Find semantic spaces with active placements
            rows = conn.execute(
                """SELECT DISTINCT s.id FROM spaces s
                JOIN cloud_placements cp ON cp.space_id = s.id
                WHERE s.mode = 'semantic' AND s.layer_id = ?
                AND cp.activation > 0.1""",
                (concept_layer_id,)
            ).fetchall()
            
            for row in rows:
                space_id = row["id"]
                self._run_space_physics(space_id, results)
    
    def _run_space_physics(self, space_id: int, results: Dict) -> None:
        """Run physics for a specific space."""
        # Get placements in this space
        placements = self.placement_repo.get_by_space(space_id)
        if not placements:
            return
        
        # Get clouds for these placements
        cloud_ids = [p.cloud_id for p in placements]
        clouds = {}
        for cid in cloud_ids:
            cloud = self.cloud_repo.get_by_id(cid)
            if cloud:
                clouds[cid] = cloud
        
        # Create physics simulation
        physics = create_space_physics(space_id, placements, clouds)
        
        # Run ticks
        updates = physics.run_ticks(self.config.physics_ticks_per_step)
        
        if updates:
            # Batch update positions
            self.placement_repo.update_positions_batch([
                type('obj', (object,), {'id': pid, 'x': x, 'y': y})()
                for pid, x, y in updates
            ])
            results["position_changes"] += len(updates)
    
    def _update_coactivation_stats(self, results: Dict) -> None:
        """Update co-activation statistics from recent activations."""
        # This is handled during activation recording
        pass
    
    def _get_response_concepts(self) -> List[Dict[str, Any]]:
        """Get concepts for API response (legacy compatibility)."""
        word_layer_id = self.get_layer_id("word_form")
        if not word_layer_id:
            return []

        clouds = self.cloud_repo.get_by_layer(word_layer_id, limit=1000)
        global_space = SpaceRepository().get_global_space(word_layer_id)
        placements = {
            p.cloud_id: p
            for p in self.placement_repo.get_by_space(global_space.id)
        } if global_space else {}

        return [
            {
                "id": c.id,
                "token": c.canonical_name,
                "position": [
                    placements[c.id].x,
                    placements[c.id].y,
                ] if c.id in placements else [400.0, 300.0],
                "mass": c.mass,
                "radius": c.radius,
                "activation": c.activation,
            }
            for c in clouds
        ]
    
    def get_space(self) -> Dict[str, Any]:
        """Get current space state."""
        return {"concepts": self._get_response_concepts(), "stats": get_stats()}
    
    def reset_space(self) -> Dict[str, Any]:
        """Reset all learning."""
        reset_space()
        return {"success": True, "concepts": [], "stats": {"concepts": 0, "total_mass": 0, "tokens": 0}}


_training_manager: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager
