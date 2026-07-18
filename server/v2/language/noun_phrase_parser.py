"""Maximal noun-phrase parsing based on morphology and local agreement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Sequence

from .models import ParsedToken
from .relation_phrase_parser import RelationPhrase, RelationPhraseParser


NOUN_POS = {"NOUN", "NPRO"}
ADJECTIVE_POS = {"ADJF", "ADJS", "PRTF", "PRTS", "NUMR"}


@dataclass
class MentionDraft:
    start: int
    end: int
    head: int
    token_indices: List[int]
    surface: str
    normalized_surface: str
    lemma: str
    features: Dict[str, Any]
    preposition: str
    attributes: List[str]
    type_token: Optional[int] = None
    owner_token: Optional[int] = None
    relation_type: Optional[str] = None
    relation_function: Optional[str] = None
    modifier_token_indices: List[int] = field(default_factory=list)
    owner_modifier_token_indices: List[int] = field(default_factory=list)
    confidence: float = .82
    evidence: List[str] = field(default_factory=list)

    @property
    def mention_type(self) -> str:
        return "apposition" if self.type_token is not None else "noun_phrase"

    def as_dict(self, tokens: Sequence[ParsedToken]) -> Dict[str, Any]:
        entity_type = (
            {
                "surface": tokens[self.type_token].surface,
                "lemma": tokens[self.type_token].lemma,
            }
            if self.type_token is not None
            else None
        )
        head = tokens[self.head]
        return {
            "token_start": self.start,
            "token_end": self.end,
            "token_indices": list(self.token_indices),
            "mention_type": self.mention_type,
            "surface": self.surface,
            "normalized_surface": self.normalized_surface,
            "head_token_index": self.head,
            "head": head.lemma,
            "head_surface": head.surface,
            "canonical_lemma": self.lemma,
            "attributes": [
                {
                    "token_index": index,
                    "surface": tokens[index].surface,
                    "lemma": tokens[index].lemma,
                }
                for index in self.modifier_token_indices
            ],
            "entity_value": {
                "surface": head.surface,
                "canonical_name": head.lemma,
            },
            "entity_type": entity_type,
            "owner": (
                {
                    "head_token_index": self.owner_token,
                    "surface": tokens[self.owner_token].surface,
                    "lemma": tokens[self.owner_token].lemma,
                    "attributes": [
                        {
                            "token_index": index,
                            "surface": tokens[index].surface,
                            "lemma": tokens[index].lemma,
                        }
                        for index in self.owner_modifier_token_indices
                    ],
                }
                if self.owner_token is not None
                else None
            ),
            "preposition": self.preposition,
            "relation_type": self.relation_type,
            "relation_function": self.relation_function,
            "grammatical_features": dict(self.features),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


class EntityMentionParser:
    """Collect maximal, non-overlapping noun phrases.

    Domain vocabulary is deliberately absent. Apposition, modification and
    genitive attachment are decided from morphology, capitalization and span
    continuity.
    """

    prepositions = {
        "в", "во", "на", "у", "к", "ко", "из", "от", "с", "со", "для",
        "через", "под", "над", "между", "около", "возле", "рядом", "после",
        "до", "перед", "по", "о", "об",
    }

    @staticmethod
    def _agrees(
        left: ParsedToken,
        right: ParsedToken,
        *,
        require_case: bool = True,
    ) -> bool:
        for key in ("case", "number", "gender"):
            if key == "case" and not require_case:
                continue
            a = left.features.get(key)
            b = right.features.get(key)
            if a and b and a != b:
                return False
        return True

    @staticmethod
    def _proper_name(token: ParsedToken) -> bool:
        if "proper_name" in token.features:
            return bool(token.features["proper_name"])
        return bool(
            token.surface[:1].isupper()
        )

    def _left_modifiers(
        self,
        tokens: Sequence[ParsedToken],
        head_index: int,
        consumed: set[int],
    ) -> List[int]:
        result: List[int] = []
        cursor = head_index - 1
        while cursor >= 0 and cursor not in consumed:
            token = tokens[cursor]
            if (
                token.pos in ADJECTIVE_POS
                and self._agrees(token, tokens[head_index])
            ):
                result.append(cursor)
                cursor -= 1
                continue
            break
        return list(reversed(result))

    def _genitive_tail(
        self,
        tokens: Sequence[ParsedToken],
        head_index: int,
        consumed: set[int],
    ) -> tuple[List[int], Optional[int]]:
        cursor = head_index + 1
        modifiers: List[int] = []
        while (
            cursor < len(tokens)
            and cursor not in consumed
            and tokens[cursor].pos in ADJECTIVE_POS
        ):
            modifiers.append(cursor)
            cursor += 1
        if (
            cursor < len(tokens)
            and cursor not in consumed
            and tokens[cursor].pos in NOUN_POS
            and tokens[cursor].grammatical_case == "gent"
            and all(
                self._agrees(tokens[index], tokens[cursor])
                for index in modifiers
            )
            and tokens[head_index].grammatical_case != "gent"
        ):
            return modifiers + [cursor], cursor
        return [], None

    def parse(
        self,
        tokens: Sequence[ParsedToken],
        relation_phrases: Sequence[RelationPhrase] = (),
        *,
        excluded_indices: Collection[int] = (),
    ) -> List[MentionDraft]:
        mentions: List[MentionDraft] = []
        # Operators that open an unfilled semantic slot are not entity
        # mentions.  They are passed in by the language pipeline rather than
        # inferred from domain vocabulary, so this parser remains reusable for
        # statements and typed questions alike.
        consumed: set[int] = set(excluded_indices)
        relation_parser = RelationPhraseParser()
        for token in tokens:
            if token.index in consumed or token.pos not in NOUN_POS:
                continue
            modifiers = self._left_modifiers(tokens, token.index, consumed)
            start = modifiers[0] if modifiers else token.index
            end = token.index
            head = token.index
            type_token: Optional[int] = None
            owner_token: Optional[int] = None
            owner_modifier_indices: List[int] = []
            evidence = ["noun_head"]
            if end + 1 < len(tokens) and end + 1 not in consumed:
                following = tokens[end + 1]
                same_slot = (
                    (
                        not token.features.get("number")
                        or not following.features.get("number")
                        or token.features.get("number")
                        == following.features.get("number")
                    )
                    and not any(
                        item.pos in {"CONJ", "PRCL"}
                        for item in tokens[end + 1:following.index]
                    )
                )
                is_proper_apposition = (
                    following.pos in NOUN_POS
                    and (
                        self._proper_name(following)
                        or following.surface[:1].isupper()
                    )
                    and not self._proper_name(token)
                    and same_slot
                    and (
                        following.grammatical_case == "nomn"
                        or following.grammatical_case == token.grammatical_case
                    )
                )
                if is_proper_apposition:
                    end = following.index
                    head = following.index
                    type_token = token.index
                    evidence.extend([
                        "common_noun_plus_proper_name",
                        "single_syntactic_span",
                    ])
            if type_token is None:
                tail, owner_token = self._genitive_tail(
                    tokens, token.index, consumed
                )
                if tail:
                    end = tail[-1]
                    owner_modifier_indices = tail[:-1]
                    evidence.append("genitive_dependency")
            indices = list(range(start, end + 1))
            values = [tokens[index] for index in indices]
            head_token = tokens[head]
            phrase_feature_token = (
                tokens[type_token]
                if type_token is not None
                else head_token
            )
            attribute_indices = list(modifiers)
            relation = relation_parser.governing(relation_phrases, start)
            preposition = ""
            relation_type = None
            relation_function = None
            if relation:
                preposition = relation.surface
                relation_type = relation.relation_type
                relation_function = relation.grammatical_function
                evidence.append("compound_relation_operator")
            else:
                cursor = start - 1
                while cursor >= 0 and start - cursor <= 3:
                    candidate = tokens[cursor]
                    if candidate.normalized in self.prepositions:
                        preposition = candidate.normalized
                        break
                    if candidate.pos not in ADJECTIVE_POS and candidate.pos != "ADVB":
                        break
                    cursor -= 1
            mentions.append(MentionDraft(
                start=start,
                end=end,
                head=head,
                token_indices=indices,
                surface=" ".join(item.surface for item in values),
                normalized_surface=" ".join(item.normalized for item in values),
                lemma=head_token.lemma,
                features=dict(phrase_feature_token.features),
                preposition=preposition,
                attributes=[tokens[index].lemma for index in attribute_indices],
                type_token=type_token,
                owner_token=owner_token,
                relation_type=relation_type,
                relation_function=relation_function,
                modifier_token_indices=attribute_indices,
                owner_modifier_token_indices=owner_modifier_indices,
                confidence=.96 if type_token is not None else .88 if modifiers else .82,
                evidence=evidence,
            ))
            consumed.update(indices)
        return mentions
