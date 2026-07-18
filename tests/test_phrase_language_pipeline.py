from pathlib import Path

import pytest

from server.v2.hive import V2HiveService
from server.v2.language import (
    EntityMentionParser,
    ParsedToken,
    UniversalLanguageAnalyzer,
)
from server.v2.repository import V2Repository
from server.v2.semantics import RoleHypothesisResolver, SpatialRelationResolver
from server.v2.training import RussianMorphology, TrainingPipelineV2


def _ask(service: V2HiveService, text: str) -> dict:
    hive_id = service.create()["hive"]["id"]
    query = service.query(hive_id, text)
    answer = service.vibration_run(hive_id, 3)["answer"]
    return {"query": query, "answer": answer}


def test_contextual_morphology_keeps_agreeing_adjective_inside_phrase():
    analysis = UniversalLanguageAnalyzer(RussianMorphology()).analyze(
        "Карта лежит в северной комнате.",
        detect_question=False,
    )

    location = next(
        mention for mention in analysis.mentions
        if mention.lemma == "комната"
    )
    adjective = analysis.tokens[3]

    assert location.surface == "северной комнате"
    assert location.preposition == "в"
    assert location.attributes == ["северный"]
    assert location.head == 4
    assert adjective.pos == "ADJF"
    assert adjective.features["case"] == "loct"
    assert any(
        item["code"] == "MORPH_ANALYSIS_AMBIGUITY"
        for item in analysis.diagnostics
    )


def test_multiple_adjectives_use_the_same_distant_noun_head():
    analysis = UniversalLanguageAnalyzer(RussianMorphology()).analyze(
        "Карта лежит в старой северной комнате.",
        detect_question=False,
    )

    location = next(
        mention for mention in analysis.mentions
        if mention.lemma == "комната"
    )

    assert location.surface == "старой северной комнате"
    assert location.attributes == ["старый", "северный"]
    assert [
        analysis.tokens[index].features["case"] for index in (3, 4, 5)
    ] == ["loct", "loct", "loct"]


def test_genitive_dependency_stays_inside_one_maximal_mention():
    analysis = UniversalLanguageAnalyzer(RussianMorphology()).analyze(
        "Руководитель большого проекта выступил.",
        detect_question=False,
    )

    mention = analysis.mentions[0].as_dict(analysis.tokens)

    assert len(analysis.mentions) == 1
    assert mention["surface"] == "Руководитель большого проекта"
    assert mention["head"] == "руководитель"
    assert mention["attributes"] == []
    assert mention["owner"]["lemma"] == "проект"
    assert [
        attribute["lemma"] for attribute in mention["owner"]["attributes"]
    ] == ["большой"]


def test_apposition_exposes_name_type_and_full_surface_separately():
    analysis = UniversalLanguageAnalyzer(RussianMorphology()).analyze(
        "Инженер Мария проверила модуль Орион.",
        detect_question=False,
    )

    mentions = [mention.as_dict(analysis.tokens) for mention in analysis.mentions]
    subject = mentions[0]
    object_ = mentions[1]

    assert subject["mention_type"] == "apposition"
    assert subject["entity_value"]["canonical_name"] == "мария"
    assert subject["entity_type"]["lemma"] == "инженер"
    assert subject["surface"] == "Инженер Мария"
    assert object_["mention_type"] == "apposition"
    assert object_["entity_value"]["canonical_name"] == "орион"
    assert object_["entity_type"]["lemma"] == "модуль"


def test_apposition_uses_type_case_for_role_when_word_order_is_inverted():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Робота Искра ремонтирует Артём."
    )
    service = V2HiveService(repository)

    result = _ask(service, "Кого ремонтирует Артём?")

    assert result["answer"]["surface_answer"] == "Искру."
    with repository.transaction() as conn:
        participant = conn.execute(
            """SELECT ep.grammatical_slot,ep.semantic_role
               FROM event_participants ep
               JOIN entities e ON e.cloud_id=ep.entity_id
               WHERE e.canonical_lemma='искра'"""
        ).fetchone()
    assert participant["grammatical_slot"] == "direct_object"
    assert participant["semantic_role"] == "patient"


def test_role_question_operator_is_not_absorbed_into_a_following_proper_name():
    analysis = UniversalLanguageAnalyzer(RussianMorphology()).analyze(
        "Кому Артём передал контейнер?"
    )

    assert analysis.question_operator is not None
    assert analysis.question_operator.token_indices == [0]
    assert [(mention.surface, mention.token_indices) for mention in analysis.mentions] == [
        ("Артём", [1]),
        ("контейнер", [3]),
    ]


def test_typed_question_is_constraint_and_uses_unfilled_construction_slot():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Оператор направил дрон Орион в дальний ангар."
    )
    service = V2HiveService(repository)

    result = _ask(
        service,
        "Какой дрон оператор направил в дальний ангар?",
    )
    frame = result["query"]["query_frame"]
    candidate = result["query"]["candidates"][0]

    assert frame["requested_slot"] == "direct_object"
    assert frame["requested_role"] == "object"
    assert frame["semantic_requested_role"] == "patient"
    assert frame["requested_role_hypotheses"][0]["role"] == "patient"
    assert frame["roles"]["agent"]["lemma"] == "оператор"
    assert frame["slot_constraints"]["object"]["is_a"]["lemma"] == "дрон"
    assert frame["question_operator"]["operator_type"] == "TYPED_ROLE_QUERY"
    assert candidate["entity_value"] == "Орион"
    assert candidate["entity_type"]["lemma"] == "дрон"
    assert candidate["full_surface"] == "дрон Орион"
    assert candidate["constraint_matches"]["taxonomy"]["passed"] is True
    assert result["answer"]["surface_answer"] == "Ориона."


def test_direction_and_compound_spatial_relation_preserve_full_mentions():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Оператор направил дрон Орион в дальний ангар. "
        "Маяк находится рядом с высокой башней."
    )
    service = V2HiveService(repository)

    direction = _ask(service, "Куда оператор направил дрон Орион?")
    near = _ask(service, "Где находится маяк?")

    assert direction["query"]["query_frame"]["requested_role"] == "destination"
    assert direction["answer"]["surface_answer"] == "В дальний ангар."
    assert near["answer"]["surface_answer"] == "Рядом с высокой башней."
    assert (
        near["query"]["candidates"][0]["mention"]["surface"]
        == "высокой башней"
    )
    with repository.transaction() as conn:
        assert conn.execute(
            """SELECT 1 FROM concept_relations
               WHERE relation_type='LOCATED_NEAR'"""
        ).fetchone() is not None


def test_construction_pattern_groups_modifiers_instead_of_tokenizing_them():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Кот ремонтирует синюю машину. "
        "Медведь ремонтирует старую машину."
    )

    with repository.transaction() as conn:
        templates = conn.execute(
            """SELECT surface_pattern,evidence_count
               FROM construction_templates
               WHERE predicate_lemma='ремонтировать'"""
        ).fetchall()

    assert len(templates) == 1
    assert templates[0]["evidence_count"] == 2
    assert "ADJ" not in templates[0]["surface_pattern"]
    assert templates[0]["surface_pattern"].count("NP:") == 2


@pytest.mark.parametrize(
    ("modifier_surface", "noun_surface", "modifier_lemma", "noun_lemma"),
    [
        ("тарная", "велума", "тарный", "велум"),
        ("севная", "норика", "севный", "норик"),
        ("кельная", "дарима", "кельный", "дарим"),
    ],
)
def test_noun_phrase_structure_does_not_depend_on_domain_vocabulary(
    modifier_surface: str,
    noun_surface: str,
    modifier_lemma: str,
    noun_lemma: str,
):
    tokens = [
        ParsedToken(
            index=0,
            surface=modifier_surface,
            normalized=modifier_surface,
            lemma=modifier_lemma,
            pos="ADJF",
            features={"case": "accs", "number": "sing", "gender": "femn"},
        ),
        ParsedToken(
            index=1,
            surface=noun_surface,
            normalized=noun_surface,
            lemma=noun_lemma,
            pos="NOUN",
            features={"case": "accs", "number": "sing", "gender": "femn"},
        ),
    ]

    mentions = EntityMentionParser().parse(tokens)

    assert len(mentions) == 1
    assert mentions[0].head == 1
    assert mentions[0].attributes == [modifier_lemma]
    assert mentions[0].surface == f"{modifier_surface} {noun_surface}"


def test_dative_participant_keeps_recipient_and_experiencer_hypotheses():
    hypotheses = RoleHypothesisResolver().hypotheses(
        "indirect_object",
        has_direct_object=True,
    )

    assert [item["role"] for item in hypotheses[:2]] == [
        "recipient",
        "experiencer",
    ]
    assert hypotheses[0]["confidence"] > hypotheses[1]["confidence"]
    assert all(item["evidence"] for item in hypotheses)


def test_spatial_resolver_combines_case_with_learned_predicate_profile():
    resolver = SpatialRelationResolver()

    motion = resolver.resolve(
        preposition="в",
        grammatical_case="accs",
        predicate_profile={"motion_score": .9, "state_score": .1},
    )
    state = resolver.resolve(
        preposition="в",
        grammatical_case="loct",
        predicate_profile={"motion_score": .1, "state_score": .9},
    )

    assert motion["destination"] > motion["location"]
    assert state["location"] > state["destination"]


def test_universal_language_modules_do_not_embed_fixture_domain_vocabulary():
    root = Path(__file__).parents[1] / "server" / "v2"
    production_sources = [
        *sorted((root / "language").glob("*.py")),
        *sorted((root / "semantics").glob("*.py")),
    ]
    fixture_lexemes = {
        "ангар", "башня", "дрон", "животное", "контейнер", "кот",
        "модуль", "пациент", "рыба", "робот",
    }

    violations = {
        path.name: sorted(
            lemma for lemma in fixture_lexemes
            if lemma in path.read_text(encoding="utf-8").casefold()
        )
        for path in production_sources
    }

    assert not {path: words for path, words in violations.items() if words}
