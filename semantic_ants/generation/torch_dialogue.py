from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from semantic_ants.learning.checkpoint import Checkpoint

try:  # pragma: no cover - import path depends on runtime image
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - training reports the missing runtime
    torch = None
    nn = None


URI_RE = re.compile(r"/[A-Za-z0-9_\-/а-яА-ЯёЁ.]+")
TERM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


@dataclass(frozen=True)
class TorchDialogueConfig:
    max_prompt_chars: int = 1400
    max_answer_chars: int = 320
    max_pairs: int = 1200
    max_sequence: int = 512
    max_context_turns: int = 8
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    train_steps: int = 8
    learning_rate: float = 0.002
    model_filename: str = "dialogue.pt"


class TorchDialogueNavigator:
    def __init__(self, config: TorchDialogueConfig | None = None) -> None:
        self.config = config or TorchDialogueConfig()

    @property
    def torch_available(self) -> bool:
        return torch is not None and nn is not None

    def model_path(self, model_dir: str | Path | None) -> Path | None:
        if model_dir is None:
            return None
        return Path(model_dir) / self.config.model_filename

    def build_prompt(
        self,
        input_text: str,
        tokens: list[str],
        activated_concepts: list[dict[str, Any]],
        routes: list[Any],
        checkpoint: Checkpoint,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        lines: list[str] = ["context:"]
        for turn in (chat_history or [])[-self.config.max_context_turns :]:
            role = str(turn.get("role", "user"))
            text = " ".join(str(turn.get("text", "")).split())
            if text:
                lines.append(f"{role}: {text}")
        lines.extend(
            [
                f"user: {' '.join(input_text.split())}",
                f"tokens: {', '.join(tokens)}",
                "concepts:",
            ]
        )
        for item in activated_concepts[:8]:
            uri = str(item.get("uri", ""))
            label = str(item.get("label", ""))
            score = item.get("score", 0.0)
            lines.append(f"- {uri} label={label} score={score}")
        lines.append("routes:")
        for route in routes[:6]:
            concepts = getattr(route, "concepts", [])
            score = getattr(route, "total_score", 0.0)
            lines.append(f"- score={score}: {' -> '.join(map(str, concepts[:6]))}")
        memory = self._memory_candidates("\n".join(lines), checkpoint, count=3)
        if memory:
            lines.append("learned_memory:")
            for answer in memory:
                lines.append(f"- {answer}")
        lines.append("assistant:")
        return "\n".join(lines)[-self.config.max_prompt_chars :]

    def generate(
        self,
        semantic_prompt: str,
        checkpoint: Checkpoint,
        model_dir: str | Path | None = None,
        fallback: str = "",
        count: int = 3,
    ) -> list[str]:
        candidates: list[str] = []
        generated = self._generate_with_torch(semantic_prompt, checkpoint, model_dir)
        if generated and self._plausible_answer(generated):
            candidates.append(generated)
        candidates.extend(self._memory_candidates(semantic_prompt, checkpoint, count=count))
        if fallback:
            candidates.append(fallback)
        return _unique_nonempty(candidates)[: max(count, 1)]

    def train_pair(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        accepted_answer: str,
        checkpoint: Checkpoint,
        reward: float = 1.0,
        model_dir: str | Path | None = None,
        steps: int | None = None,
    ) -> None:
        self.train_pairs(
            [
                {
                    "stimulus": stimulus,
                    "prompt": semantic_prompt,
                    "concepts": concepts,
                    "answer": accepted_answer,
                    "reward": reward,
                }
            ],
            checkpoint,
            model_dir=model_dir,
            steps=steps,
        )

    def train_pairs(
        self,
        pairs: list[dict[str, Any]],
        checkpoint: Checkpoint,
        model_dir: str | Path | None = None,
        steps: int | None = None,
    ) -> None:
        training_pairs: list[dict[str, str]] = []
        stored_pairs = checkpoint.mini_generator.setdefault("dialogue_pairs", [])
        for pair in pairs:
            stimulus = str(pair.get("stimulus", ""))
            semantic_prompt = str(pair.get("prompt", ""))
            accepted_answer = str(pair.get("answer", ""))
            concepts = [str(value) for value in pair.get("concepts", [])]
            reward = float(pair.get("reward", 1.0))
            if not accepted_answer:
                continue
            clipped_prompt = semantic_prompt[: self.config.max_prompt_chars]
            clipped_answer = accepted_answer[: self.config.max_answer_chars]
            checkpoint.remember_accepted_answer(
                stimulus=stimulus,
                semantic_prompt=clipped_prompt,
                concepts=concepts,
                answer=clipped_answer,
                reward=reward,
                limit=self.config.max_pairs,
            )
            stored_pairs.append(
                {
                    "stimulus": stimulus,
                    "prompt": clipped_prompt,
                    "answer": clipped_answer,
                    "concepts": list(dict.fromkeys(concepts)),
                    "reward": reward,
                    "created_at": time.time(),
                }
            )
            training_pairs.append({"prompt": clipped_prompt, "answer": clipped_answer})
        del stored_pairs[:-self.config.max_pairs]
        checkpoint.mini_generator["dialogue_config"] = asdict(self.config)
        self._train_torch_pairs(training_pairs, checkpoint, model_dir, steps=steps)

    def train_negative(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        rejected_answer: str,
        checkpoint: Checkpoint,
        reason: str = "",
    ) -> None:
        if not rejected_answer:
            return
        checkpoint.remember_negative(
            stimulus=stimulus,
            semantic_prompt=semantic_prompt[: self.config.max_prompt_chars],
            concepts=concepts,
            answer=rejected_answer[: self.config.max_answer_chars],
            reason=reason,
        )

    def _train_torch_pairs(
        self,
        pairs: list[dict[str, str]],
        checkpoint: Checkpoint,
        model_dir: str | Path | None,
        steps: int | None = None,
    ) -> None:
        if not pairs:
            return
        model_path = self.model_path(model_dir)
        if model_path is None:
            return
        if not self.torch_available:
            raise RuntimeError("PyTorch is required for dialogue training")
        texts = [_training_text(pair["prompt"], pair["answer"]) for pair in pairs]
        vocab = _ensure_vocab(checkpoint, "".join(texts))
        if len(vocab) < 4:
            return
        model = _TinyCausalTransformer(
            vocab_size=len(vocab),
            d_model=self.config.d_model,
            heads=self.config.heads,
            layers=self.config.layers,
            max_sequence=self.config.max_sequence,
        )
        self._load_torch_state(model, model_path)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        step_count = steps if steps is not None else self.config.train_steps
        for _ in range(max(step_count, 1)):
            for text in texts:
                ids = _encode(text, vocab)[-self.config.max_sequence :]
                if len(ids) < 3:
                    continue
                source = torch.tensor([ids[:-1]], dtype=torch.long)
                target = torch.tensor([ids[1:]], dtype=torch.long)
                optimizer.zero_grad()
                logits = model(source)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), target.reshape(-1))
                loss.backward()
                optimizer.step()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "config": asdict(self.config),
                "saved_at": time.time(),
            },
            model_path,
        )

    def _generate_with_torch(
        self,
        semantic_prompt: str,
        checkpoint: Checkpoint,
        model_dir: str | Path | None,
    ) -> str | None:
        if not self.torch_available:
            return None
        model_path = self.model_path(model_dir)
        if model_path is None or not model_path.exists():
            return None
        package = _torch_load(model_path)
        vocab = package.get("vocab") if isinstance(package, dict) else None
        if not isinstance(vocab, list) or len(vocab) < 4:
            vocab = checkpoint.mini_generator.get("dialogue_vocab")
        if not isinstance(vocab, list) or len(vocab) < 4:
            return None
        model = _TinyCausalTransformer(
            vocab_size=len(vocab),
            d_model=self.config.d_model,
            heads=self.config.heads,
            layers=self.config.layers,
            max_sequence=self.config.max_sequence,
        )
        self._load_torch_state(model, model_path)
        model.eval()
        prompt = semantic_prompt[-self.config.max_prompt_chars :]
        ids = _encode(prompt, vocab)[-self.config.max_sequence :]
        generated: list[int] = []
        for _ in range(self.config.max_answer_chars):
            source_ids = ids[-self.config.max_sequence :]
            source = torch.tensor([source_ids], dtype=torch.long)
            with torch.no_grad():
                next_id = int(torch.argmax(model(source)[0, -1]).item())
            ids.append(next_id)
            generated.append(next_id)
            char = vocab[next_id]
            if char == "\0":
                break
            if char == "\n" and len(generated) > 8:
                break
        return _decode(generated, vocab).replace("\0", "").strip() or None

    def _load_torch_state(self, model: Any, model_path: Path) -> None:
        if not model_path.exists():
            return
        package = _torch_load(model_path)
        if not isinstance(package, dict):
            return
        raw_state = package.get("state_dict", package)
        if not isinstance(raw_state, dict):
            return
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in raw_state.items()
            if key in current and hasattr(value, "shape") and tuple(value.shape) == tuple(current[key].shape)
        }
        current.update(compatible)
        model.load_state_dict(current)

    def _memory_candidates(self, semantic_prompt: str, checkpoint: Checkpoint, count: int) -> list[str]:
        prompt_concepts = set(URI_RE.findall(semantic_prompt))
        prompt_terms = _terms(semantic_prompt)
        scored: list[tuple[float, str]] = []
        for item in checkpoint.accepted_answers:
            answer = str(item.get("answer", ""))
            if not answer:
                continue
            stored_prompt = str(item.get("semantic_prompt", ""))
            stimulus = str(item.get("stimulus", ""))
            concepts = set(map(str, item.get("concepts", [])))
            overlap = len(prompt_concepts & concepts)
            lexical_terms = _terms(stored_prompt) | _terms(stimulus)
            lexical = len(prompt_terms & lexical_terms) / max(len(lexical_terms), 1)
            reward = float(item.get("reward", 0.0))
            score = overlap * 2.0 + lexical + reward
            if score > 0:
                scored.append((score, answer))
        scored.sort(key=lambda value: value[0], reverse=True)
        return [answer for _, answer in scored[:count]]

    def _plausible_answer(self, value: str) -> bool:
        clean = " ".join(value.split())
        if len(clean) < 2:
            return False
        if any(marker in clean.lower() for marker in ("concepts:", "routes:", "assistant:", "tokens:")):
            return False
        counts: dict[str, int] = {}
        for char in clean:
            counts[char] = counts.get(char, 0) + 1
        if counts and max(counts.values()) / len(clean) > 0.45:
            return False
        if len(set(TERM_RE.findall(clean.lower()))) <= 1 and len(clean) > 24:
            return False
        return True


if nn is not None:  # pragma: no cover - exercised through integration tests

    class _TinyCausalTransformer(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, heads: int, layers: int, max_sequence: int) -> None:
            super().__init__()
            self.max_sequence = max_sequence
            self.token = nn.Embedding(vocab_size, d_model)
            self.position = nn.Embedding(max_sequence, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, ids: Any) -> Any:
            length = ids.size(1)
            positions = torch.arange(length, device=ids.device).unsqueeze(0)
            hidden = self.token(ids) + self.position(positions)
            mask = torch.triu(torch.ones(length, length, device=ids.device), diagonal=1).bool()
            return self.output(self.encoder(hidden, mask=mask))

else:  # pragma: no cover - unavailable runtime

    class _TinyCausalTransformer:  # type: ignore[no-redef]
        pass


def _training_text(prompt: str, answer: str) -> str:
    return f"{prompt}\n{answer}\0"


def _ensure_vocab(checkpoint: Checkpoint, text: str) -> list[str]:
    vocab = checkpoint.mini_generator.get("dialogue_vocab")
    if not isinstance(vocab, list) or not vocab:
        vocab = ["\0", "\n", " ", "?"]
    for char in text:
        if char not in vocab:
            vocab.append(char)
    checkpoint.mini_generator["dialogue_vocab"] = vocab
    return vocab


def _encode(text: str, vocab: list[str]) -> list[int]:
    unknown = vocab.index("?") if "?" in vocab else 0
    return [vocab.index(char) if char in vocab else unknown for char in text]


def _decode(ids: list[int], vocab: list[str]) -> str:
    return "".join(vocab[index] for index in ids if 0 <= index < len(vocab))


def _terms(text: str) -> set[str]:
    return set(TERM_RE.findall(text.lower()))


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(str(value).split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location="cpu")
