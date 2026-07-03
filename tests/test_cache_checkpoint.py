import tempfile
import unittest
from pathlib import Path

from semantic_ants.core.models import SemanticEdge
from semantic_ants.learning.checkpoint import Checkpoint, CheckpointStore
from semantic_ants.providers.cache import JsonCache


class CacheCheckpointTest(unittest.TestCase):
    def test_json_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonCache(tmp)
            cache.set("key", {"value": 1})
            self.assertEqual(cache.get("key"), {"value": 1})

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            store = CheckpointStore(path)
            checkpoint = Checkpoint()
            checkpoint.reinforce_edge("/c/en/a", "RelatedTo", "/c/en/b")
            store.save(checkpoint)
            loaded = store.load()
            edge = SemanticEdge("/c/en/a", "/c/en/b", "RelatedTo")
            self.assertGreater(loaded.pheromone_for(edge), 1.0)


if __name__ == "__main__":
    unittest.main()
