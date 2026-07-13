"""Training manager for the relation-free concept field."""

import time
from collections import Counter
from typing import Any, Dict, List, Optional

from .database import ensure_concepts, get_concepts, get_stats, init_db, reset_space, update_concepts
from .physics import ConceptState, PhysicsConfig, compute_center, concept_radius, place_new_concepts, run_simulation
from .tokenizer import split_sentences, tokenize


class TrainingManager:
    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        init_db()

    def _states(self) -> List[ConceptState]:
        return [
            ConceptState(
                id=int(item["id"]),
                token=item["token"],
                position=item["position"],
                mass=float(item["mass"]),
                radius=concept_radius(float(item["mass"])),
            )
            for item in get_concepts()
        ]

    @staticmethod
    def _serialize(states: List[ConceptState]) -> List[Dict[str, Any]]:
        return [
            {
                "id": state.id,
                "token": state.token,
                "position": [round(float(state.position[0]), 4), round(float(state.position[1]), 4)],
                "mass": round(float(state.mass), 4),
                "radius": round(float(concept_radius(state.mass)), 4),
                "activation": round(float(state.activation), 4),
            }
            for state in sorted(states, key=lambda item: (-item.mass, item.id))
        ]

    def learn(self, text: str) -> Dict[str, Any]:
        started = time.time()
        raw_sentences = split_sentences(text)
        sentences = [tokenize(sentence) for sentence in raw_sentences]
        sentences = [sentence for sentence in sentences if sentence]
        tokens = [token for sentence in sentences for token in sentence]
        if not tokens:
            return {"success": False, "concepts": [], "stats": get_stats(), "time_ms": 0, "error": "No valid tokens found"}

        current_states = self._states()
        center = compute_center(current_states)
        existing_tokens = {state.token for state in current_states}
        ensure_concepts(list(dict.fromkeys(tokens)), center)
        states = self._states()
        new_states = [state for state in states if state.token not in existing_tokens]
        known_states = [state for state in states if state.token in existing_tokens]
        place_new_concepts(new_states, known_states, self.config)

        counts = Counter(tokens)
        total_tokens = max(1, len(tokens))
        for state in states:
            state.activation = counts.get(state.token, 0) / total_tokens

        active_states = [state for state in states if state.activation > 0]
        context_position = compute_center(active_states)
        run_simulation(states, sentences, self.config, context_position)
        update_concepts((state.id, state.position, state.mass) for state in states)

        return {
            "success": True,
            "concepts": self._serialize(states),
            "stats": {**get_stats(), "tokens": len(tokens)},
            "time_ms": int((time.time() - started) * 1000),
        }

    def get_space(self) -> Dict[str, Any]:
        return {"concepts": self._serialize(self._states()), "stats": get_stats()}

    def reset_space(self) -> Dict[str, Any]:
        reset_space()
        return {"success": True, "concepts": [], "stats": {"concepts": 0, "total_mass": 0, "tokens": 0}}


_training_manager: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager
