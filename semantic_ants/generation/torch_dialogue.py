from __future__ import annotations

import re
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from semantic_ants.generation.sentences import render_concept_pattern
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
        lang: str | None = None,
    ) -> str:
        lines: list[str] = ["context:"]
        for turn in (chat_history or [])[-self.config.max_context_turns :]:
            role = str(turn.get("role", "user"))
            text = " ".join(str(turn.get("text", "")).split())
            if text:
                lines.append(f"{role}: {text}")
        lines.extend(
            [
                f"lang: {lang or 'auto'}",
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
        memory = self._memory_candidates("\n".join(lines), checkpoint, count=3, lang=lang)
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
        fallback: str | list[str] = "",
        count: int = 3,
        lang: str | None = None,
    ) -> list[str]:
        candidates: list[str] = []
        candidates.extend(self._memory_candidates(semantic_prompt, checkpoint, count=count, lang=lang))
        generated = self._generate_with_torch(semantic_prompt, checkpoint, model_dir)
        if generated and self._plausible_answer(generated) and _language_matches(generated, lang):
            candidates.append(generated)
        if isinstance(fallback, list):
            candidates.extend(fallback)
        elif fallback:
            candidates.append(fallback)
        candidates = [candidate for candidate in candidates if _language_matches(candidate, lang)]
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
        answer_lang: str | None = None,
        source_lang: str | None = None,
    ) -> None:
        self.train_pairs(
            [
                {
                    "stimulus": stimulus,
                    "prompt": semantic_prompt,
                    "concepts": concepts,
                    "answer": accepted_answer,
                    "reward": reward,
                    "answer_lang": answer_lang,
                    "source_lang": source_lang,
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
        stored_pairs = checkpoint.mini_generator.setdefault("dialogue_patterns", [])
        for pair in pairs:
            stimulus = str(pair.get("stimulus", ""))
            semantic_prompt = str(pair.get("prompt", ""))
            accepted_answer = str(pair.get("answer", ""))
            concepts = [str(value) for value in pair.get("concepts", [])]
            reward = float(pair.get("reward", 1.0))
            answer_lang = str(pair.get("answer_lang") or pair.get("response_lang") or pair.get("target_lang") or "")
            source_lang = str(pair.get("source_lang") or pair.get("lang") or "")
            normalized_answer_lang = answer_lang if answer_lang in {"ru", "en"} else ""
            normalized_source_lang = source_lang if source_lang in {"ru", "en"} else ""
            if not accepted_answer:
                continue
            clipped_prompt = semantic_prompt[: self.config.max_prompt_chars]
            clipped_answer = accepted_answer[: self.config.max_answer_chars]
            pattern = checkpoint.remember_accepted_answer(
                stimulus=stimulus,
                semantic_prompt=clipped_prompt,
                concepts=concepts,
                answer=clipped_answer,
                reward=reward,
                limit=self.config.max_pairs,
                lang=normalized_answer_lang or None,
                source_lang=normalized_source_lang or None,
            )
            if isinstance(pattern, dict):
                stored_pairs.append(
                    {
                        "concepts": list(dict.fromkeys(concepts)),
                        "answer_concepts": list(pattern.get("answer_concepts", [])),
                        "answer": clipped_answer,
                        "lang": pattern.get("lang", "auto"),
                        "source_lang": pattern.get("source_lang", normalized_source_lang or ""),
                        "reward": reward,
                        "created_at": time.time(),
                    }
                )
            training_pairs.append({"prompt": clipped_prompt, "answer": clipped_answer})
        del stored_pairs[:-self.config.max_pairs]
        checkpoint.mini_generator["dialogue_patterns"] = stored_pairs
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
        answer_lang: str | None = None,
        source_lang: str | None = None,
    ) -> None:
        if not rejected_answer:
            return
        checkpoint.remember_negative(
            stimulus=stimulus,
            semantic_prompt=semantic_prompt[: self.config.max_prompt_chars],
            concepts=concepts,
            answer=rejected_answer[: self.config.max_answer_chars],
            reason=reason,
            lang=answer_lang,
            source_lang=source_lang,
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
            checkpoint.mini_generator["torch_unavailable"] = True
            return
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

    def _memory_candidates(
        self,
        semantic_prompt: str,
        checkpoint: Checkpoint,
        count: int,
        lang: str | None = None,
    ) -> list[str]:
        prompt_concepts = set(URI_RE.findall(semantic_prompt))
        input_terms = _input_terms(semantic_prompt)
        prompt_terms = set(input_terms)
        rejected = _negative_answers(semantic_prompt, checkpoint)
        scored: list[tuple[float, str]] = []
        raw_scored: list[tuple[float, str]] = []
        exact_scored: list[tuple[float, str]] = []
        exact_raw_scored: list[tuple[float, str]] = []
        memory_items = [
            *checkpoint.accepted_answers,
            *checkpoint.response_memory.values(),
        ]
        for item in memory_items:
            if not isinstance(item, dict):
                continue
            answer_concepts = [str(value) for value in item.get("answer_concepts", []) if value]
            concepts = [str(value) for value in item.get("concepts", []) if value]
            if not answer_concepts and not concepts:
                continue
            source_concepts = set(concepts)
            answer_concept_set = set(answer_concepts)
            overlap = len(prompt_concepts & source_concepts)
            answer_overlap = len(prompt_concepts & answer_concept_set)
            lexical_terms = _item_stimulus_terms(item)
            lexical_overlap = len(prompt_terms & lexical_terms)
            lexical = lexical_overlap / max(len(lexical_terms), 1)
            stimulus_clean = _clean_text(str(item.get("stimulus", "")))
            exact_stimulus = bool(stimulus_clean and _contains_term_sequence(input_terms, stimulus_clean))
            if not (overlap or answer_overlap or exact_stimulus):
                continue
            reward = min(float(item.get("reward", item.get("weight", 0.0))), 3.0)
            score = overlap + answer_overlap * 0.5 + lexical * 2.0 + lexical_overlap * 2.0 + reward
            if exact_stimulus:
                score += 20.0
            item_lang = str(item.get("lang") or lang or "auto")
            answers = _item_answers(item)
            has_raw_answer = bool(answers)
            if not has_raw_answer:
                answers = render_concept_pattern(
                    answer_concepts or concepts,
                    checkpoint,
                    lang=item_lang if item_lang in {"ru", "en"} else None,
                    reward=reward,
                    count=max(1, min(count, 3)),
                )
            for answer in answers:
                if score > 0 and answer and _clean_text(answer) not in rejected:
                    if exact_stimulus and has_raw_answer:
                        target = exact_raw_scored
                    elif exact_stimulus:
                        target = exact_scored
                    elif has_raw_answer:
                        target = raw_scored
                    else:
                        target = scored
                    target.append((score, answer))
        selected = exact_raw_scored or exact_scored or raw_scored or scored
        selected.sort(key=lambda value: value[0], reverse=True)
        return _unique_nonempty([answer for _, answer in selected[:count]])

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


def _language_matches(value: str, lang: str | None) -> bool:
    if lang == "ru":
        return any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in value)
    if lang == "en":
        return not any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in value)
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


def _clean_text(text: str) -> str:
    return " ".join(TERM_RE.findall(text.lower()))


def _input_terms(text: str) -> list[str]:
    match = re.search(r'"input_text":\s*"((?:\\.|[^"\\])*)"', text)
    if match:
        try:
            return list(TERM_RE.findall(json.loads(f'"{match.group(1)}"').lower()))
        except json.JSONDecodeError:
            return list(TERM_RE.findall(match.group(1).lower()))
    for line in text.splitlines():
        if line.startswith("user:"):
            return list(TERM_RE.findall(line.split(":", 1)[1].lower()))
    return []


def _contains_term_sequence(haystack: list[str], clean_sequence: str) -> bool:
    needle = clean_sequence.split()
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def _item_answers(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("answer", "response"):
        value = " ".join(str(item.get(key, "")).split())
        if value:
            values.append(value)
    return _unique_nonempty(values)


def _item_stimulus_terms(item: dict[str, Any]) -> set[str]:
    terms = _terms(str(item.get("stimulus", "")))
    terms.update(_input_terms(str(item.get("semantic_prompt", ""))))
    return terms


def _negative_answers(semantic_prompt: str, checkpoint: Checkpoint) -> set[str]:
    prompt_concepts = set(URI_RE.findall(semantic_prompt))
    prompt_terms = set(_input_terms(semantic_prompt))
    rejected: set[str] = set()
    for item in checkpoint.negative_memory:
        if not isinstance(item, dict):
            continue
        concepts = {str(value) for value in item.get("concepts", []) if value}
        lexical_terms = _item_stimulus_terms(item)
        if not (prompt_concepts & concepts or prompt_terms & lexical_terms):
            continue
        for answer in _item_answers(item):
            rejected.add(_clean_text(answer))
    return rejected


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
