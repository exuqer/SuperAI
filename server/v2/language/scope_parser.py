"""Operator scope parsing for negation, modality and quantification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .models import Modality, ParsedToken, Polarity


MODAL_LEMMAS = {
    "мочь": Modality.CAN,
    "можно": Modality.MAY,
    "должный": Modality.MUST,
    "должен": Modality.MUST,
    "следовать": Modality.SHOULD,
    "хотеть": Modality.WANT,
    "намереваться": Modality.INTEND,
    "собираться": Modality.INTEND,
    "пытаться": Modality.TRY,
    "верить": Modality.BELIEVE,
    "считать": Modality.BELIEVE,
    "знать": Modality.KNOW,
}
FREQUENCY_LEMMAS = {
    "всегда",
    "часто",
    "редко",
    "иногда",
    "никогда",
    "обычно",
}
QUANTITY_POS = {"NUMR"}
PARTICIPANT_POS = {"NOUN", "NPRO"}
PREDICATE_POS = {"VERB", "INFN", "PRTS", "GRND"}
ATTRIBUTE_POS = {"ADJF", "ADJS", "PRTF", "PRTS"}


class ScopeParser:
    def parse_modality(self, tokens: Sequence[ParsedToken]) -> Optional[Modality]:
        for token in tokens:
            normalized = token.normalized.casefold()
            lemma = token.lemma.casefold()
            if lemma in MODAL_LEMMAS:
                return MODAL_LEMMAS[lemma]
            if normalized in MODAL_LEMMAS:
                return MODAL_LEMMAS[normalized]
        return None

    def parse_negation(
        self,
        tokens: Sequence[ParsedToken],
        *,
        token_offset: int = 0,
    ) -> tuple[Polarity, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        negation_indices = [
            index for index, token in enumerate(tokens)
            if token.normalized.casefold() in {"не", "ни"}
        ]
        if not negation_indices:
            return Polarity.POSITIVE, None, []
        predicate_index = next(
            (
                index for index, token in enumerate(tokens)
                if token.pos in PREDICATE_POS
            ),
            None,
        )
        scopes: List[Dict[str, Any]] = []
        for negation_index in negation_indices:
            target_index = negation_index + 1
            if target_index >= len(tokens):
                scopes.append({
                    "scope_type": "CLAUSE",
                    "target": None,
                    "negation_token_index": token_offset + negation_index,
                    "confidence": 0.72,
                    "evidence": ["clause_final_negation"],
                })
                continue
            target = tokens[target_index]
            target_normalized = target.normalized.casefold()
            target_lemma = target.lemma.casefold()
            if target_lemma in FREQUENCY_LEMMAS or target_normalized in FREQUENCY_LEMMAS:
                scope_type = "FREQUENCY"
                target_value: Any = target_lemma
                evidence = "negation_before_frequency"
            elif target_lemma in MODAL_LEMMAS or target_normalized in MODAL_LEMMAS:
                scope_type = "MODALITY"
                target_value = (
                    MODAL_LEMMAS.get(target_lemma)
                    or MODAL_LEMMAS.get(target_normalized)
                ).value
                evidence = "negation_before_modal"
            elif target.pos in QUANTITY_POS or target_normalized.isdigit():
                scope_type = "QUANTITY"
                target_value = target.surface
                evidence = "negation_before_quantity"
            elif target.pos in ATTRIBUTE_POS:
                scope_type = "ATTRIBUTE"
                target_value = target.lemma
                evidence = "negation_before_attribute"
            elif target.pos in PARTICIPANT_POS:
                scope_type = "PARTICIPANT" if (
                    predicate_index is not None and target_index < predicate_index
                ) else "PARTICIPANT_VALUE"
                target_value = {
                    "lemma": target.lemma,
                    "surface": target.surface,
                    "token_index": token_offset + target_index,
                    "observation_signature": {
                        **({
                            f"morph:case:{target.features['case']}": 0.9,
                        } if target.features.get("case") else {}),
                        (
                            "position:before_predicate"
                            if (
                                predicate_index is not None
                                and target_index < predicate_index
                            )
                            else "position:after_predicate"
                        ): 0.8,
                    },
                }
                evidence = (
                    "preverbal_participant_negation"
                    if scope_type == "PARTICIPANT"
                    else "argument_value_negation"
                )
            elif target.pos in PREDICATE_POS:
                scope_type = "EVENT"
                target_value = target.lemma
                evidence = "predicate_event_negation"
            else:
                scope_type = "CLAUSE"
                target_value = target.lemma
                evidence = "untyped_negation_target"
            scope = {
                "scope_type": scope_type,
                "target": target_value,
                "negation_token_index": token_offset + negation_index,
                "target_token_index": token_offset + target_index,
                "confidence": 0.96 if scope_type != "CLAUSE" else 0.72,
                "evidence": [evidence],
            }
            if scope_type == "PARTICIPANT_VALUE":
                alternative_index = next(
                    (
                        index for index in range(target_index + 1, len(tokens))
                        if tokens[index].normalized.casefold() == "а"
                    ),
                    None,
                )
                if alternative_index is not None and alternative_index + 1 < len(tokens):
                    alternative = tokens[alternative_index + 1]
                    scope["asserted_alternative"] = {
                        "lemma": alternative.lemma,
                        "surface": alternative.surface,
                        "token_index": token_offset + alternative_index + 1,
                    }
            scopes.append(scope)
        primary = max(
            scopes,
            key=lambda item: (
                item["confidence"],
                item["scope_type"] != "CLAUSE",
            ),
        )
        if len(scopes) > 1:
            primary = {**primary, "additional_scopes": scopes}
        return Polarity.NEGATIVE, primary, scopes

    def parse(
        self,
        tokens: Sequence[ParsedToken],
        *,
        token_offset: int = 0,
    ) -> Dict[str, Any]:
        polarity, negation_scope, all_scopes = self.parse_negation(
            tokens,
            token_offset=token_offset,
        )
        modality = self.parse_modality(tokens)
        quantifiers = [
            {
                "token_index": token_offset + index,
                "surface": token.surface,
                "lemma": token.lemma,
            }
            for index, token in enumerate(tokens)
            if token.pos in QUANTITY_POS
            or token.normalized.casefold() in {
                "все",
                "каждый",
                "любой",
                "несколько",
                "много",
                "мало",
            }
        ]
        return {
            "polarity": polarity,
            "negation_scope": negation_scope,
            "negation_scopes": all_scopes,
            "modality": modality,
            "quantifiers": quantifiers,
        }
