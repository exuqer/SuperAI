from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from semantic_ants.learning.checkpoint import Checkpoint

if os.environ.get("SEMANTIC_ANTS_USE_TORCH", "0") == "1":  # pragma: no cover - optional runtime path
    try:
        import torch
        import torch.nn as nn
    except Exception:
        torch = None
        nn = None
else:  # pragma: no cover - optional dependency
    torch = None
    nn = None


URI_RE = re.compile(r"/[A-Za-z0-9_\-/а-яА-ЯёЁ.]+")


@dataclass(frozen=True)
class MiniLLMConfig:
    max_prompt_chars: int = 1200
    max_answer_chars: int = 320
    max_pairs: int = 500
    max_sequence: int = 256
    d_model: int = 32
    heads: int = 4
    layers: int = 1
    torch_steps: int = int(os.environ.get("SEMANTIC_ANTS_TORCH_STEPS", "0"))


class MiniTransformerSpeechModule:
    """Маленький речевой модуль поверх принятой смысловой памяти."""

    def __init__(self, config: MiniLLMConfig | None = None) -> None:
        self.config = config or MiniLLMConfig()

    @property
    def torch_available(self) -> bool:
        return torch is not None and nn is not None

    def generate(
        self,
        semantic_prompt: str,
        checkpoint: Checkpoint,
        fallback: str = "",
        count: int = 3,
    ) -> list[str]:
        candidates: list[str] = []
        candidates.extend(self._memory_candidates(semantic_prompt, checkpoint, count=count))
        generated = self._generate_with_torch(semantic_prompt, checkpoint)
        if generated:
            candidates.append(generated)
        if fallback:
            candidates.append(fallback)
        if not candidates:
            candidates.append(self._prompt_fallback(semantic_prompt))
        return _unique_nonempty(candidates)[: max(count, 1)]

    def train_pair(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        accepted_answer: str,
        checkpoint: Checkpoint,
        reward: float = 1.0,
    ) -> None:
        if not accepted_answer:
            return
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
        pairs = checkpoint.mini_generator.setdefault("training_pairs", [])
        pairs.append(
            {
                "prompt": clipped_prompt,
                "answer": clipped_answer,
                "concepts": list(dict.fromkeys(concepts)),
                "reward": float(reward),
            }
        )
        del pairs[:-self.config.max_pairs]
        self._train_torch_pair(clipped_prompt, clipped_answer, checkpoint)

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

    def _memory_candidates(self, semantic_prompt: str, checkpoint: Checkpoint, count: int) -> list[str]:
        prompt_concepts = set(URI_RE.findall(semantic_prompt))
        prompt_terms = _terms(semantic_prompt)
        scored: list[tuple[float, str]] = []
        for item in checkpoint.accepted_answers:
            answer = str(item.get("answer", ""))
            if not answer:
                continue
            concepts = set(map(str, item.get("concepts", [])))
            stored_prompt = str(item.get("semantic_prompt", ""))
            overlap = len(prompt_concepts & concepts)
            lexical = len(prompt_terms & _terms(stored_prompt)) / max(len(prompt_terms), 1)
            reward = float(item.get("reward", 0.0))
            score = overlap * 2.0 + lexical + reward
            if score > 0:
                scored.append((score, answer))
        scored.sort(key=lambda value: value[0], reverse=True)
        return [answer for _, answer in scored[:count]]

    def _prompt_fallback(self, semantic_prompt: str) -> str:
        concepts = URI_RE.findall(semantic_prompt)
        if concepts:
            labels = ", ".join(_label_from_uri(value) for value in concepts[:3])
            return f"Смысловой маршрут указывает на: {labels}."
        return "Пока не хватает принятых ответов для речевого модуля; использую графовый смысловой слой."

    def _train_torch_pair(self, semantic_prompt: str, accepted_answer: str, checkpoint: Checkpoint) -> None:
        if not self.torch_available or self.config.torch_steps <= 0:
            return
        text = f"{semantic_prompt}\n=>\n{accepted_answer}"
        vocab = _ensure_vocab(checkpoint, text)
        if len(vocab) < 4:
            return
        ids = _encode(text, vocab)[-self.config.max_sequence :]
        if len(ids) < 3:
            return
        model = _TinyTransformerLM(
            vocab_size=len(vocab),
            d_model=self.config.d_model,
            heads=self.config.heads,
            layers=self.config.layers,
        )
        raw_state = checkpoint.mini_generator.get("torch_state")
        if isinstance(raw_state, dict):
            try:
                model.load_state_dict({key: torch.tensor(value) for key, value in raw_state.items()}, strict=False)
            except Exception:
                pass
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        source = torch.tensor([ids[:-1]], dtype=torch.long)
        target = torch.tensor([ids[1:]], dtype=torch.long)
        for _ in range(self.config.torch_steps):
            optimizer.zero_grad()
            logits = model(source)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), target.reshape(-1))
            loss.backward()
            optimizer.step()
        checkpoint.mini_generator["torch_vocab"] = vocab
        checkpoint.mini_generator["torch_state"] = {
            key: value.detach().cpu().tolist() for key, value in model.state_dict().items()
        }

    def _generate_with_torch(self, semantic_prompt: str, checkpoint: Checkpoint) -> str | None:
        if not self.torch_available:
            return None
        raw_state = checkpoint.mini_generator.get("torch_state")
        vocab = checkpoint.mini_generator.get("torch_vocab")
        if not isinstance(raw_state, dict) or not isinstance(vocab, list) or len(vocab) < 4:
            return None
        try:
            model = _TinyTransformerLM(
                vocab_size=len(vocab),
                d_model=self.config.d_model,
                heads=self.config.heads,
                layers=self.config.layers,
            )
            model.load_state_dict({key: torch.tensor(value) for key, value in raw_state.items()}, strict=False)
            model.eval()
            ids = _encode(f"{semantic_prompt}\n=>\n", vocab)[-self.config.max_sequence :]
            for _ in range(self.config.max_answer_chars):
                source = torch.tensor([ids[-self.config.max_sequence :]], dtype=torch.long)
                with torch.no_grad():
                    next_id = int(torch.argmax(model(source)[0, -1]).item())
                ids.append(next_id)
                char = vocab[next_id]
                if char in {"\n", "\0"} and len(ids) > 8:
                    break
            decoded = _decode(ids, vocab).split("=>", 1)[-1].strip()
            return decoded[: self.config.max_answer_chars] or None
        except Exception:
            return None


if nn is not None:  # pragma: no cover - optional dependency path

    class _TinyTransformerLM(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, heads: int, layers: int) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, d_model)
            self.position = nn.Embedding(512, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, ids: Any) -> Any:
            positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            hidden = self.token(ids) + self.position(positions)
            mask = torch.triu(torch.ones(ids.size(1), ids.size(1), device=ids.device), diagonal=1).bool()
            return self.output(self.encoder(hidden, mask=mask))

else:  # pragma: no cover - optional dependency path

    class _TinyTransformerLM:  # type: ignore[no-redef]
        pass


def _ensure_vocab(checkpoint: Checkpoint, text: str) -> list[str]:
    vocab = checkpoint.mini_generator.get("torch_vocab")
    if not isinstance(vocab, list) or not vocab:
        vocab = ["\0", "\n", " ", "?"]
    for char in text:
        if char not in vocab:
            vocab.append(char)
    checkpoint.mini_generator["torch_vocab"] = vocab
    return vocab


def _encode(text: str, vocab: list[str]) -> list[int]:
    unknown = vocab.index("?") if "?" in vocab else 0
    return [vocab.index(char) if char in vocab else unknown for char in text]


def _decode(ids: list[int], vocab: list[str]) -> str:
    return "".join(vocab[index] for index in ids if 0 <= index < len(vocab))


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\wа-яА-ЯёЁ]+", text.lower()))


def _label_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1].replace("_", " ")


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
