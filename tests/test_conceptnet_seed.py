from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_ants.knowledge.base import bootstrap_builtin_knowledge
from semantic_ants.knowledge.conceptnet_seed import bootstrap_conceptnet_knowledge
from semantic_ants.learning.checkpoint import Checkpoint


class ConceptNetSeedTest(unittest.TestCase):
    def test_offline_builtin_bootstrap_does_not_touch_conceptnet(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Checkpoint()
            with patch("semantic_ants.knowledge.conceptnet_seed._dump_stream", side_effect=AssertionError("network call")):
                report = bootstrap_builtin_knowledge(checkpoint, allow_network=False)
            self.assertTrue(report.changed)
            self.assertEqual(checkpoint.metadata.get("conceptnet_seed_version"), None)

    def test_conceptnet_seed_links_existing_and_adds_concepts(self):
        checkpoint = Checkpoint()
        checkpoint.remember_concept_label("/c/en/sun", "sun")
        checkpoint.aliases["sun"] = "/c/en/sun"

        dump_text = (
            "/a/[test]/r/RelatedTo/c/en/sun/c/en/light\t/r/RelatedTo\t/c/en/sun\t/c/en/light\t"
            '{"dataset":"/d/conceptnet/4/en","license":"cc:by-sa/4.0","weight":1.0,"surfaceText":"[[sun]] is related to [[light]]"}\n'
            "/a/[test]/r/RelatedTo/c/en/light/c/en/heat\t/r/RelatedTo\t/c/en/light\t/c/en/heat\t"
            '{"dataset":"/d/conceptnet/4/en","license":"cc:by-sa/4.0","weight":1.0,"surfaceText":"[[light]] is related to [[heat]]"}\n'
            "/a/[test]/r/RelatedTo/c/en/sun/c/en/star\t/r/RelatedTo\t/c/en/sun\t/c/en/star\t"
            '{"dataset":"/d/conceptnet/4/en","license":"cc:by-sa/4.0","weight":1.0,"surfaceText":"[[sun]] is related to [[star]]"}\n'
        )

        with patch("semantic_ants.knowledge.conceptnet_seed._dump_stream", return_value=io.BytesIO(dump_text.encode("utf-8"))):
            report = bootstrap_conceptnet_knowledge(checkpoint, allow_network=True)

        self.assertTrue(report.changed)
        self.assertGreaterEqual(report.concepts, 2)
        self.assertGreaterEqual(report.edges, 3)
        self.assertTrue(any(edge["relation"] == "RelatedTo" for edge in checkpoint.custom_edges))
        self.assertIn("/c/en/light", checkpoint.metadata["concept_definitions"])
        self.assertIn("/c/en/star", checkpoint.metadata["concept_definitions"])
        self.assertEqual(checkpoint.metadata["concept_definitions"]["/c/en/light"]["label"], "light")


if __name__ == "__main__":
    unittest.main()
