"""Tests for SuperAI training module"""
import pytest
import tempfile
import os
from pathlib import Path

# Add the semantic_ants to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_ants.tokenizer import tokenize, split_sentences, normalize_text
from semantic_ants.physics import WordState, PhysicsConfig, run_simulation, place_new_words_around_center


def test_tokenize_russian_and_latin():
    """Test tokenization of Russian and Latin words."""
    text = "Привет, как дела? Hello world!"
    tokens = tokenize(text)
    assert "привет" in tokens
    assert "как" in tokens
    assert "дела" in tokens
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_ignores_punctuation():
    """Test that punctuation is filtered out."""
    text = "Привет! Как дела? Всё отлично."
    tokens = tokenize(text)
    assert "!" not in tokens
    assert "?" not in tokens
    assert "." not in tokens
    assert "," not in tokens


def test_split_sentences():
    """Test sentence splitting."""
    text = "Привет. Как дела? Всё отлично!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert "Привет" in sentences[0]
    assert "Как дела" in sentences[1]
    assert "Всё отлично" in sentences[2]


def test_normalize_text():
    """Test text normalization."""
    assert normalize_text("  Привет   мир  ") == "Привет мир"
    assert normalize_text("\n\r\tПривет\nмир\t") == "Привет мир"
    assert normalize_text("") == ""


def test_physics_word_state():
    """Test WordState physics calculations."""
    w1 = WordState(word="test1", mass=1.0, x=100, y=100)
    w2 = WordState(word="test2", mass=2.0, x=200, y=200)
    
    dist = w1.distance_to(w2)
    assert dist > 0
    assert abs(dist - 141.42) < 0.1


def test_physics_simulation_runs():
    """Test that physics simulation runs without errors."""
    config = PhysicsConfig(steps=5)
    words = [
        WordState(word="word1", mass=1.0, x=100, y=100),
        WordState(word="word2", mass=2.0, x=200, y=200),
        WordState(word="word3", mass=1.5, x=300, y=100),
    ]
    phrase_groups = [words[:2]]  # First two words in a phrase
    
    run_simulation(words, phrase_groups, config)
    
    # Check that positions were updated
    for w in words:
        assert w.x != 0 or w.y != 0
        assert abs(w.x) < 2000  # Within bounds
        assert abs(w.y) < 2000


def test_place_new_words_around_center():
    """Test placing new words around center."""
    config = PhysicsConfig()
    existing = [
        WordState(word="e1", mass=1.0, x=400, y=300),
        WordState(word="e2", mass=1.0, x=600, y=400),
    ]
    new_words = [
        WordState(word="n1", mass=1.0, x=0, y=0),
        WordState(word="n2", mass=1.0, x=0, y=0),
    ]
    
    place_new_words_around_center(new_words, existing, config)
    
    # New words should be placed around the center of existing words
    for w in new_words:
        assert w.x > 0 and w.x < 1000
        assert w.y > 0 and w.y < 700


def test_mass_increment():
    """Test that existing word mass increments by 0.1."""
    # This test is more of an integration test - we test it through the database
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])