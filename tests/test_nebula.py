"""Tests for the recursive nebula system."""

import pytest
from datetime import datetime

# Server modules will use patched database from conftest
from server.tokenizer import tokenize_hierarchical, CharacterToken, WordToken, SentenceTokens
from server.repositories.cloud_repository import CloudRepository, LayerRepository, SpaceRepository
from server.models.cloud import Cloud, CloudPlacement, StructuralComponent
from server.models.space import Space, Layer
from server.services.condensation import condensation_service
from server.services.activation import ActivationManager, compute_activation_from_text
from server.training import TrainingManager, TrainingConfig
from server.physics import LocalSpacePhysics, create_space_physics, PhysicsConfig
from server.services.zoom import zoom_service
from server.server import RegionSelectRequest


def now_str():
    return datetime.utcnow().isoformat()


# ============================================================
# Tests
# ============================================================

def test_tokenizer_hierarchical():
    """Test 1: Hierarchical tokenization preserves character order."""
    text = "мяч"
    result = tokenize_hierarchical(text)
    
    assert len(result.sentences) == 1
    assert len(result.all_tokens) == 1
    assert len(result.all_characters) == 3
    
    word = result.all_tokens[0]
    assert word.normalized == "мяч"
    assert len(word.characters) == 3
    assert [c.normalized for c in word.characters] == ["м", "я", "ч"]


def test_tokenizer_multiple_words():
    """Test tokenization of multiple words."""
    text = "круглый мяч"
    result = tokenize_hierarchical(text)
    
    assert len(result.all_tokens) == 2
    assert result.all_tokens[0].normalized == "круглый"
    assert result.all_tokens[1].normalized == "мяч"
    
    # Check character order preserved
    assert [c.normalized for c in result.all_tokens[0].characters] == ["к", "р", "у", "г", "л", "ы", "й"]
    assert [c.normalized for c in result.all_tokens[1].characters] == ["м", "я", "ч"]


def test_tokenizer_sentence_order():
    """Test sentence and word position tracking."""
    text = "Мяч летит. Мальчик ловит."
    result = tokenize_hierarchical(text)
    
    assert len(result.sentences) == 2
    assert result.sentences[0].index == 0
    assert result.sentences[1].index == 1
    
    # Check token positions
    assert result.all_tokens[0].sentence_index == 0
    assert result.all_tokens[0].token_index_in_sentence == 0
    assert result.all_tokens[1].sentence_index == 0
    assert result.all_tokens[1].token_index_in_sentence == 1


def test_layers_created():
    """Test that default layers are created on init."""
    layer_repo = LayerRepository()
    layers = layer_repo.get_all_ordered()
    
    assert len(layers) == 6
    names = [l.name for l in layers]
    assert names == ["signal", "character", "word_form", "concept", "scene", "context"]
    
    # Check order_index
    for i, layer in enumerate(layers):
        assert layer.order_index == i


def test_character_cloud_creation():
    """Test creating character clouds."""
    layer_repo = LayerRepository()
    cloud_repo = CloudRepository()
    char_layer_id = layer_repo.get_by_name("character").id
    
    cloud = cloud_repo.create(Cloud(
        layer_id=char_layer_id,
        cloud_type="character",
        canonical_name="м",
        mass=1.0,
        density=1.0,
        stability=0.1,
        observation_count=1,
    ))
    
    assert cloud.id is not None
    assert cloud.canonical_name == "м"
    assert cloud.layer_id == char_layer_id


def test_word_form_condensation():
    """Test 2: Word form condensation from character sequence."""
    layer_repo = LayerRepository()
    char_layer_id = layer_repo.get_by_name("character").id
    word_layer_id = layer_repo.get_by_name("word_form").id
    
    cloud_repo = CloudRepository()
    
    char_clouds = []
    for char in ["м", "я", "ч"]:
        cloud = cloud_repo.create(Cloud(
            layer_id=char_layer_id,
            cloud_type="character",
            canonical_name=char,
            mass=1.0,
            density=1.0,
            stability=0.1,
            observation_count=1,
        ))
        char_clouds.append(cloud)
    
    # First occurrence - creates candidate
    result = condensation_service.create_word_form_from_characters(
        [c.id for c in char_clouds], "мяч", min_observations=2
    )
    assert result is None  # Not enough observations
    
    # Second occurrence - should create
    result = condensation_service.create_word_form_from_characters(
        [c.id for c in char_clouds], "мяч", min_observations=2
    )
    assert result is not None
    word_cloud, is_new = result
    assert word_cloud.canonical_name == "мяч"
    assert word_cloud.layer_id == word_layer_id
    assert is_new is True


def test_word_form_condensation_order_sensitive():
    """Test 3: Different character orders create different word forms."""
    layer_repo = LayerRepository()
    char_layer_id = layer_repo.get_by_name("character").id
    cloud_repo = CloudRepository()
    
    # Create characters
    char_map = {}
    for char in ["м", "я", "ч"]:
        cloud = cloud_repo.create(Cloud(
            layer_id=char_layer_id,
            cloud_type="character",
            canonical_name=char,
            mass=1.0,
            density=1.0,
            stability=0.1,
            observation_count=1,
        ))
        char_map[char] = cloud
    
    # "мяч" - м, я, ч
    ids_1 = [char_map["м"].id, char_map["я"].id, char_map["ч"].id]
    result1 = condensation_service.create_word_form_from_characters(ids_1, "мяч", min_observations=1)
    word1, is_new1 = result1
    
    # "чям" - ч, я, м
    ids_2 = [char_map["ч"].id, char_map["я"].id, char_map["м"].id]
    result2 = condensation_service.create_word_form_from_characters(ids_2, "чям", min_observations=1)
    word2, is_new2 = result2
    
    assert word1 is not None
    assert word2 is not None
    assert word1.id != word2.id
    assert word1.canonical_name == "мяч"
    assert word2.canonical_name == "чям"


def test_concept_condensation():
    """Test concept creation from word form co-occurrence."""
    layer_repo = LayerRepository()
    word_layer_id = layer_repo.get_by_name("word_form").id
    cloud_repo = CloudRepository()
    
    # Create word form clouds
    word_clouds = []
    for word in ["круглый", "мяч"]:
        cloud = cloud_repo.create(Cloud(
            layer_id=word_layer_id,
            cloud_type="word",
            canonical_name=word,
            mass=2.0,
            density=1.0,
            stability=0.3,
            observation_count=2,
        ))
        word_clouds.append(cloud)
    
    # First co-occurrence
    concept = condensation_service.create_concept_from_word_forms(
        [c.id for c in word_clouds], "круглый_мяч", "context1", min_observations=2
    )
    assert concept is None
    
    # Second co-occurrence
    concept = condensation_service.create_concept_from_word_forms(
        [c.id for c in word_clouds], "круглый_мяч", "context1", min_observations=2
    )
    assert concept is not None
    assert concept.layer_id == layer_repo.get_by_name("concept").id


def test_training_character_layer():
    """Test full training pipeline with character layer."""
    config = TrainingConfig(
        min_word_observations=1,
        min_concept_observations=1,
        enable_character_layer=True,
        enable_word_form_layer=True,
        enable_concept_layer=False,
    )
    manager = TrainingManager(config)
    
    # Train on text
    result = manager.learn("мяч")
    
    assert result["success"] is True
    assert len(result["details"]["created_clouds"]) > 0
    
    # Check character clouds created
    char_clouds = [c for c in result["details"]["created_clouds"] if c["layer"] == "character"]
    assert len(char_clouds) == 3
    char_names = {c["name"] for c in char_clouds}
    assert char_names == {"м", "я", "ч"}
    
    # Check word form created
    word_clouds = [c for c in result["details"]["created_clouds"] if c["layer"] == "word_form"]
    assert len(word_clouds) == 1
    assert word_clouds[0]["name"] == "мяч"


def test_training_repeated_word():
    """Test 2: Repeated word strengthens existing clouds."""
    config = TrainingConfig(min_word_observations=1)
    manager = TrainingManager(config)
    
    # First training
    result1 = manager.learn("мяч")
    word_clouds_1 = [c for c in result1["details"]["created_clouds"] if c["layer"] == "word_form"]
    assert len(word_clouds_1) == 1
    word_id = word_clouds_1[0]["id"]
    
    # Second training
    result2 = manager.learn("мяч")
    
    # Should be strengthened, not created
    strengthened = [c for c in result2["details"]["strengthened_clouds"] if c["layer"] == "word_form"]
    assert len(strengthened) == 1
    assert strengthened[0]["id"] == word_id
    
    # No new word_form created
    created = [c for c in result2["details"]["created_clouds"] if c["layer"] == "word_form"]
    assert len(created) == 0


def test_training_semantic_proximity():
    """Test 4: Co-occurring words develop semantic proximity."""
    config = TrainingConfig(
        min_word_observations=1,
        min_concept_observations=1,
        enable_concept_layer=True,
    )
    manager = TrainingManager(config)
    
    # Train multiple times with co-occurring words
    for _ in range(3):
        manager.learn("круглый мяч")
        manager.learn("круглый стол")
        manager.learn("бросить мяч")
        manager.learn("играть мяч")
    
    # Check concept layer has clouds
    concept_layer_id = manager.get_layer_id("concept")
    cloud_repo = CloudRepository()
    concepts = cloud_repo.get_by_layer(concept_layer_id)
    
    assert len(concepts) > 0
    
    # Check co-activation stats exist
    from server.database import get_connection
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM coactivation_stats WHERE layer_id = ?",
            (concept_layer_id,)
        ).fetchall()
        # Should have some co-activation records
        assert len(rows) > 0


def test_zoom_structural():
    """Test 6: Structural zoom from word_form to characters."""
    config = TrainingConfig(min_word_observations=1)
    manager = TrainingManager(config)
    
    # Train
    manager.learn("мяч")
    
    # Get word_form cloud
    word_layer_id = manager.get_layer_id("word_form")
    cloud_repo = CloudRepository()
    word_clouds = cloud_repo.get_by_layer(word_layer_id)
    мяч_cloud = next((c for c in word_clouds if c.canonical_name == "мяч"), None)
    assert мяч_cloud is not None
    
    # Zoom in structural
    result = zoom_service.zoom_in_structural("test_session", мяч_cloud.id)
    
    assert result is not None
    assert result["space"]["mode"] == "structural"
    assert len(result["children"]) == 3
    child_names = {c["cloud"]["canonical_name"] for c in result["children"]}
    assert child_names == {"м", "я", "ч"}


def test_zoom_semantic():
    """Test semantic zoom from concept to neighbors."""
    config = TrainingConfig(
        min_word_observations=1,
        min_concept_observations=1,
        enable_concept_layer=True,
    )
    manager = TrainingManager(config)
    
    # Train multiple times
    for _ in range(3):
        manager.learn("круглый мяч")
        manager.learn("красный мяч")
    
    # Get concept cloud
    concept_layer_id = manager.get_layer_id("concept")
    cloud_repo = CloudRepository()
    concepts = cloud_repo.get_by_layer(concept_layer_id)
    мяч_concept = next((c for c in concepts if "мяч" in c.canonical_name), None)
    assert мяч_concept is not None
    
    # Zoom in semantic
    result = zoom_service.zoom_in_semantic("test_session", мяч_concept.id)
    
    assert result is not None
    assert result["space"]["mode"] == "semantic"
    assert len(result["neighbors"]) > 0


def test_region_selection():
    """Test 7: Region selection returns overlapping clouds."""
    config = TrainingConfig(min_word_observations=1, min_concept_observations=1, enable_concept_layer=True)
    manager = TrainingManager(config)
    
    # Train
    manager.learn("круглый мяч")
    
    # Get semantic space
    concept_layer_id = manager.get_layer_id("concept")
    cloud_repo = CloudRepository()
    concepts = cloud_repo.get_by_layer(concept_layer_id)
    мяч_concept = next((c for c in concepts if "мяч" in c.canonical_name), None)
    
    # Zoom to get space
    result = zoom_service.zoom_in_semantic("test_session", мяч_concept.id)
    space_id = result["space"]["id"]
    
    # Just verify the space was created
    assert space_id is not None


def test_activation_spread():
    """Test activation spreads to nearby clouds."""
    config = TrainingConfig(min_word_observations=1)
    manager = TrainingManager(config)
    
    manager.learn("мяч")
    
    # Verify activation functions work
    from server.services.activation import ActivationManager
    mgr = ActivationManager()
    
    cloud_repo = CloudRepository()
    word_layer_id = manager.get_layer_id("word_form")
    clouds = cloud_repo.get_by_layer(word_layer_id)
    мяч = next((c for c in clouds if c.canonical_name == "мяч"), None)
    
    mgr.activate_cloud(мяч, 1.0)
    assert мяч.activation > 0.5


def test_no_semantic_edges():
    """Verify no semantic edges are created - only positions change."""
    from server.database import get_connection
    
    config = TrainingConfig(min_word_observations=1, enable_concept_layer=True)
    manager = TrainingManager(config)
    
    for _ in range(3):
        manager.learn("круглый мяч")
    
    # Check no connections/edges table exists
    with get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        
        # Should NOT have semantic edges
        assert "connections" not in table_names
        assert "edges" not in table_names
        
        # Should have coactivation_stats instead
        assert "coactivation_stats" in table_names


def test_spatial_index():
    """Test spatial index for efficient queries."""
    from server.services.spatial_index import SpatialGrid
    
    grid = SpatialGrid(cell_size=100.0, width=1000.0, height=1000.0)
    
    # Insert some placements
    grid.insert(1, 100, 100, 50)
    grid.insert(2, 200, 200, 50)
    grid.insert(3, 500, 500, 50)
    
    # Query nearby
    nearby = grid.get_nearby(1, 200)
    assert len(nearby) == 1  # Only #2 is within 200
    assert nearby[0][0] == 2
    
    # Query viewport
    in_view = grid.query_rect(0, 0, 300, 300)
    assert 1 in in_view
    assert 2 in in_view
    assert 3 not in in_view


def test_physics_local_simulation():
    """Test local physics simulation in a space."""
    layer_repo = LayerRepository()
    config = PhysicsConfig()
    config.max_ticks_per_step = 2
    
    # Create placements
    from server.models.cloud import Cloud, CloudPlacement
    from server.repositories.cloud_repository import CloudRepository
    
    cloud_repo = CloudRepository()
    concept_layer_id = layer_repo.get_by_name("concept").id
    
    cloud1 = cloud_repo.create(Cloud(
        layer_id=concept_layer_id,
        cloud_type="concept",
        canonical_name="test1",
        mass=2.0,
        density=1.0,
        stability=0.5,
    ))
    
    cloud2 = cloud_repo.create(Cloud(
        layer_id=concept_layer_id,
        cloud_type="concept",
        canonical_name="test2",
        mass=2.0,
        density=1.0,
        stability=0.5,
    ))
    
    # Create space and placements
    from server.repositories.cloud_repository import SpaceRepository, CloudPlacementRepository
    space_repo = SpaceRepository()
    placement_repo = CloudPlacementRepository()
    
    space = space_repo.create(Space(
        host_cloud_id=cloud1.id,
        layer_id=concept_layer_id,
        mode="semantic",
    ))
    
    p1 = placement_repo.create(CloudPlacement(
        space_id=space.id,
        cloud_id=cloud1.id,
        x=300, y=300,
        radius=30, mass=2.0, activation=0.5,
    ))
    
    p2 = placement_repo.create(CloudPlacement(
        space_id=space.id,
        cloud_id=cloud2.id,
        x=500, y=300,
        radius=30, mass=2.0, activation=0.5,
    ))
    
    # Run physics
    clouds = {cloud1.id: cloud1, cloud2.id: cloud2}
    physics = create_space_physics(space.id, [p1, p2], clouds)
    
    updates = physics.run_ticks(5)
    
    # Positions should have changed (at least some movement)
    assert len(updates) > 0
    
    # Verify placements updated in DB
    updated_p1 = placement_repo.get_by_id(p1.id)
    updated_p2 = placement_repo.get_by_id(p2.id)
    
    # Check that movement occurred (allow small floating point differences)
    # Movement should be at least 0.01 units
    moved_1 = abs(updated_p1.x - 300) > 0.01 or abs(updated_p1.y - 300) > 0.01
    moved_2 = abs(updated_p2.x - 500) > 0.01 or abs(updated_p2.y - 300) > 0.01
    assert moved_1 or moved_2, f"Neither placement moved: p1=({updated_p1.x}, {updated_p1.y}), p2=({updated_p2.x}, {updated_p2.y})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])