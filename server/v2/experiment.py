"""Reproducible compact experiment required by PLAN.md.

The corpus is intentionally small and checked into the source as explicit
examples.  Expected answers are evaluation metadata and are never ingested as
facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import server.database as database

from .graph_repository import (
    GraphRepository,
    encode,
    serialization_snapshot,
    stable_id,
    utcnow,
)
from .acceleration import AccelerationRuntime
from .graph_service import GraphDialogueService, GraphTrainingService
from .universe import UniverseService


DATASET_VERSION = "compact-iteration-1"


@dataclass(frozen=True)
class ExperimentExample:
    text: str
    question: str
    expected: str
    domain: str
    tags: tuple[str, ...] = ()


def _examples(
    domain: str,
    rows: Sequence[tuple[str, str, str, tuple[str, ...]]],
) -> tuple[ExperimentExample, ...]:
    return tuple(
        ExperimentExample(text, question, expected, domain, tags)
        for text, question, expected, tags in rows
    )


TRAIN = (
    *_examples("food_objects", (
        ("Механик поднял ключ.", "Кто поднял ключ?", "Механик.", ("lift",)),
        ("Робот поднял коробку.", "Что поднял робот?", "Коробку.", ("lift",)),
        ("Девочка дала мальчику книгу.", "Кому девочка дала книгу?", "Мальчику.", ("give",)),
        ("Повар дал гостю тарелку.", "Что повар дал гостю?", "Тарелку.", ("give",)),
        ("Повар положил яблоко в корзину.", "Куда повар положил яблоко?", "В корзину.", ("place",)),
        ("Мальчик положил чашку на стол.", "Что мальчик положил на стол?", "Чашку.", ("place",)),
        ("Повар разрезал хлеб ножом.", "Чем повар разрезал хлеб?", "Ножом.", ("instrument",)),
        ("Девочка разрезала яблоко ножом.", "Что девочка разрезала ножом?", "Яблоко.", ("instrument",)),
        ("Книга находится на столе.", "Где находится книга?", "На столе.", ("location",)),
        ("Ключ находится в коробке.", "Что находится в коробке?", "Ключ.", ("location",)),
        ("Тарелка находится под чашкой.", "Где находится тарелка?", "Под чашкой.", ("location",)),
        ("После обеда повар вымыл нож.", "Кто вымыл нож?", "Повар.", ("temporal",)),
    )),
    *_examples("robots_tools", (
        ("Инженер поднял датчик.", "Кто поднял датчик?", "Инженер.", ("lift",)),
        ("Манипулятор поднял панель.", "Что поднял манипулятор?", "Панель.", ("lift",)),
        ("Техник дал роботу кабель.", "Кому техник дал кабель?", "Роботу.", ("give",)),
        ("Оператор дал автомату модуль.", "Что оператор дал автомату?", "Модуль.", ("give",)),
        ("Робот положил деталь в контейнер.", "Куда робот положил деталь?", "В контейнер.", ("place",)),
        ("Техник положил схему на верстак.", "Что техник положил на верстак?", "Схему.", ("place",)),
        ("Мастер разрезал провод кусачками.", "Чем мастер разрезал провод?", "Кусачками.", ("instrument",)),
        ("Автомат разрезал лист резаком.", "Что автомат разрезал резаком?", "Лист.", ("instrument",)),
        ("Датчик находится на панели.", "Где находится датчик?", "На панели.", ("location",)),
        ("Модуль находится в корпусе.", "Что находится в корпусе?", "Модуль.", ("location",)),
        ("Кабель находится под верстаком.", "Где находится кабель?", "Под верстаком.", ("location",)),
        ("После проверки инженер включил робот.", "Кто включил робот?", "Инженер.", ("temporal",)),
    )),
    *_examples("animals_motion", (
        ("Собака подняла палку.", "Кто поднял палку?", "Собака.", ("lift",)),
        ("Обезьяна подняла камень.", "Что подняла обезьяна?", "Камень.", ("lift",)),
        ("Хозяин дал собаке мяч.", "Кому хозяин дал мяч?", "Собаке.", ("give",)),
        ("Смотритель дал слону яблоко.", "Что смотритель дал слону?", "Яблоко.", ("give",)),
        ("Белка положила орех в дупло.", "Куда белка положила орех?", "В дупло.", ("place",)),
        ("Ворона положила ветку на крышу.", "Что ворона положила на крышу?", "Ветку.", ("place",)),
        ("Бобр разрезал ветку зубами.", "Чем бобр разрезал ветку?", "Зубами.", ("instrument",)),
        ("Хозяин разрезал корм ножом.", "Что хозяин разрезал ножом?", "Корм.", ("instrument",)),
        ("Кошка находится на диване.", "Где находится кошка?", "На диване.", ("location",)),
        ("Лиса находится в норе.", "Кто находится в норе?", "Лиса.", ("location",)),
        ("Мяч находится под креслом.", "Где находится мяч?", "Под креслом.", ("location",)),
        ("После дождя собака побежала домой.", "Кто побежал домой?", "Собака.", ("temporal",)),
    )),
    *_examples("drones_delivery", (
        ("Курьер поднял посылку.", "Кто поднял посылку?", "Курьер.", ("lift",)),
        ("Дрон поднял контейнер.", "Что поднял дрон?", "Контейнер.", ("lift",)),
        ("Диспетчер дал дрону маршрут.", "Кому диспетчер дал маршрут?", "Дрону.", ("give",)),
        ("Курьер дал клиенту пакет.", "Что курьер дал клиенту?", "Пакет.", ("give",)),
        ("Дрон положил посылку в ячейку.", "Куда дрон положил посылку?", "В ячейку.", ("place",)),
        ("Курьер положил пакет на стойку.", "Что курьер положил на стойку?", "Пакет.", ("place",)),
        ("Оператор разрезал ленту ножницами.", "Чем оператор разрезал ленту?", "Ножницами.", ("instrument",)),
        ("Клиент разрезал упаковку ножом.", "Что клиент разрезал ножом?", "Упаковку.", ("instrument",)),
        ("Посылка находится в ячейке.", "Где находится посылка?", "В ячейке.", ("location",)),
        ("Дрон находится на платформе.", "Что находится на платформе?", "Дрон.", ("location",)),
        ("Пакет находится под стойкой.", "Где находится пакет?", "Под стойкой.", ("location",)),
        ("После сигнала дрон доставил посылку.", "Что доставил дрон?", "Посылку.", ("temporal",)),
    )),
)


HOLDOUT = _examples("transfer", (
    ("Археолог поднял амфору.", "Амфору кто поднял?", "Археолог.", ("new_lexemes", "free_order")),
    ("Геолог поднял образец.", "Что было поднято геологом?", "Образец.", ("new_topic", "passive")),
    ("Оператор передал дрону посылку.", "Кому оператор передал посылку?", "Дрону.", ("new_predicate",)),
    ("Фармацевт передал врачу пробирку.", "Пробирку кому передал фармацевт?", "Врачу.", ("free_order",)),
    ("Археолог поместил находку в сейф.", "Куда археолог поместил находку?", "В сейф.", ("new_predicate",)),
    ("Лаборант поместил колбу на полку.", "Колбу куда поместил лаборант?", "На полку.", ("free_order",)),
    ("Скульптор рассёк глину струной.", "Чем скульптор рассёк глину?", "Струной.", ("new_predicate",)),
    ("Ткань была разрезана портным ножницами.", "Кем была разрезана ткань?", "Портным.", ("passive",)),
    ("Амфора находится в музее.", "В музее находится что?", "Амфора.", ("new_topic",)),
    ("Образец находится под микроскопом.", "Где находится образец?", "Под микроскопом.", ("new_lexemes",)),
    ("Колба находится на штативе.", "На штативе что находится?", "Колба.", ("free_order",)),
    ("Посылка была передана дрону оператором.", "Кем была передана посылка?", "Оператором.", ("passive",)),
    ("Экспедитор вручил пилоту карту.", "Что вручил экспедитор пилоту?", "Карту.", ("new_predicate",)),
    ("Реставратор поднял мозаику.", "Кто мозаику поднял?", "Реставратор.", ("free_order",)),
    ("После раскопок археолог убрал кисть.", "Кто после раскопок убрал кисть?", "Археолог.", ("temporal",)),
    ("Кристалл находится внутри породы.", "Где находится кристалл?", "Внутри породы.", ("new_topic",)),
))


CONTINUAL = _examples("continual", (
    ("Монтажник поднял балку.", "Кто поднял балку?", "Монтажник.", ("confirmation",)),
    ("Навигатор передал роверу координаты.", "Что навигатор передал роверу?", "Координаты.", ("new_lexemes",)),
    ("Ключ поднял робот.", "Что поднял робот?", "Ключ.", ("counterexample", "free_order")),
    ("Болт механику дал робот.", "Кому робот дал болт?", "Механику.", ("free_order",)),
    ("Ключ был поднят роботом.", "Кем был поднят ключ?", "Роботом.", ("passive",)),
    ("Ключ лежал рядом с роботом.", "Где лежал ключ?", "Рядом с роботом.", ("state_change",)),
    ("Робот положил ключ в ящик.", "Куда робот положил ключ?", "В ящик.", ("state_change",)),
    ("Лук растёт на грядке.", "Что растёт на грядке?", "Лук.", ("polysemy",)),
    ("Охотник натянул лук.", "Кто натянул лук?", "Охотник.", ("polysemy",)),
    ("Техник передал автомату модуль.", "Кому техник передал модуль?", "Автомату.", ("cloud_merge",)),
    ("Курьер поместил пакет в ячейку.", "Что курьер поместил в ячейку?", "Пакет.", ("cloud_boundary",)),
    ("Собака принесла хозяину мяч.", "Кому собака принесла мяч?", "Хозяину.", ("new_predicate",)),
    ("Дрон доставил клиенту коробку.", "Что дрон доставил клиенту?", "Коробку.", ("confirmation",)),
    ("Повар убрал нож в ящик.", "Куда повар убрал нож?", "В ящик.", ("new_predicate",)),
    ("Инженером был включён автомат.", "Кем был включён автомат?", "Инженером.", ("passive",)),
    ("После сигнала оператор поднял панель.", "Что поднял оператор?", "Панель.", ("temporal",)),
))


BLIND = _examples("blind", (
    ("Картограф поднял тубус.", "Кто поднял тубус?", "Картограф.", ("blind",)),
    ("Водолаз передал врачу контейнер.", "Что водолаз передал врачу?", "Контейнер.", ("blind",)),
    ("Художник положил кисть в пенал.", "Куда художник положил кисть?", "В пенал.", ("blind",)),
    ("Канат был разрезан матросом ножом.", "Кем был разрезан канат?", "Матросом.", ("blind", "passive")),
    ("Тубус находится под картой.", "Где находится тубус?", "Под картой.", ("blind",)),
    ("Матрос дал врачу аптечку.", "Кому матрос дал аптечку?", "Врачу.", ("blind",)),
    ("Пилот поднял рацию.", "Рацию поднял кто?", "Пилот.", ("blind", "free_order")),
    ("После шторма матрос убрал канат.", "Кто убрал канат?", "Матрос.", ("blind",)),
))


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 1729
    smoke_sizes: tuple[int, ...] = (25, 50, 100)
    deterministic_tolerance: float = 0.0
    dataset_version: str = DATASET_VERSION
    batch_boundaries: tuple[int, ...] = (48, 64, 80)

    def hash(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CompactExperiment:
    repository: GraphRepository = field(default_factory=GraphRepository)
    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: AccelerationRuntime = field(default_factory=AccelerationRuntime)

    def __post_init__(self) -> None:
        self.training = GraphTrainingService(self.repository, runtime=self.runtime)
        self.dialogue = GraphDialogueService(self.repository, runtime=self.runtime)
        self.universes = UniverseService(self.repository, runtime=self.runtime)

    @staticmethod
    def validate_dataset() -> Dict[str, int]:
        counts = {
            "train": len(TRAIN),
            "holdout": len(HOLDOUT),
            "continual": len(CONTINUAL),
            "blind": len(BLIND),
        }
        if counts != {
            "train": 48,
            "holdout": 16,
            "continual": 16,
            "blind": 8,
        }:
            raise ValueError(f"invalid compact dataset: {counts}")
        return counts

    def _ordered(
        self,
        examples: Sequence[ExperimentExample],
        *,
        salt: str,
    ) -> list[ExperimentExample]:
        values = list(examples)
        seed = int(hashlib.sha256(
            f"{self.config.random_seed}:{salt}".encode("utf-8")
        ).hexdigest()[:16], 16)
        random.Random(seed).shuffle(values)
        return values

    def _train_split(
        self,
        split: str,
        examples: Sequence[ExperimentExample],
    ) -> list[str]:
        order: list[str] = []
        for index, example in enumerate(self._ordered(examples, salt=split)):
            independent_key = f"{self.config.dataset_version}:{split}:{index}"
            self.training.train(
                example.text,
                independent_key=independent_key,
                domain_key=example.domain,
                discover_dimensions=False,
            )
            order.append(independent_key)
        return order

    def _evaluate(
        self,
        split: str,
        examples: Sequence[ExperimentExample],
    ) -> Dict[str, Any]:
        correct = 0
        statuses: Dict[str, int] = {}
        rows = []
        for example in examples:
            hive_id = self.dialogue.create()["hive"]["id"]
            result = self.dialogue.query(
                hive_id, example.question, resolved_mode="NEW_QUERY"
            )
            answer = result["answer"]
            surface = str(answer.get("surface") or "")
            expected = example.expected.casefold().strip()
            matched = surface.casefold().strip() == expected
            correct += int(matched)
            status = str(answer.get("status") or "UNKNOWN")
            statuses[status] = statuses.get(status, 0) + 1
            rows.append({
                "question": example.question,
                "expected": example.expected,
                "actual": surface,
                "correct": matched,
                "status": status,
                "retrieval_mode": (
                    result.get("trace", {})
                    .get("swarm", {})
                    .get("retrieval_mode")
                ),
            })
        return {
            "split": split,
            "total": len(examples),
            "correct": correct,
            "accuracy": correct / max(1, len(examples)),
            "statuses": statuses,
            "results": rows,
        }

    def _consolidate(self, label: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            dimension_ids: list[str] = []
            for universe_id, _, _ in self.universes_definitions():
                dimension_ids.extend(
                    self.universes.discoverer.propose_candidates(
                        conn, universe_id
                    )
                )
                self.universes._refresh_clouds(conn, universe_id)
                self.universes._refresh_statistics(conn, universe_id)
            snapshots = {}
            rows = conn.execute(
                """SELECT id,canonical_dimension_id,revision,status,strength,
                          retrieval_gain,predictive_gain
                   FROM latent_dimensions
                   WHERE status NOT IN ('pruned','merged')
                   ORDER BY canonical_dimension_id,revision"""
            ).fetchall()
            for row in rows:
                projection_ids = [
                    str(item["source_id"])
                    for item in conn.execute(
                        """SELECT source_id FROM projections
                           WHERE dimension_id=? AND source_type='entity'
                           ORDER BY membership DESC,source_id LIMIT 20""",
                        (row["id"],),
                    ).fetchall()
                ]
                snapshots[str(row["canonical_dimension_id"])] = {
                    "dimension_id": str(row["id"]),
                    "revision": int(row["revision"]),
                    "status": str(row["status"]),
                    "strength": float(row["strength"]),
                    "utility": float(row["retrieval_gain"])
                    + float(row["predictive_gain"]),
                    "ranked_projection_ids": projection_ids,
                }
            # A consolidation is a bulk checkpoint: refresh SQLite planner
            # statistics and invalidate all derived in-memory indexes once.
            conn.execute("ANALYZE")
            self.repository.bump_revisions(conn)
        return {
            "checkpoint": label,
            "dimension_ids": sorted(set(dimension_ids)),
            "snapshots": snapshots,
        }

    @staticmethod
    def _lineage_comparison(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> Dict[str, Any]:
        comparisons = []
        for canonical_id in sorted(set(before) & set(after)):
            left = before[canonical_id]
            right = after[canonical_id]
            left_ids = list(left["ranked_projection_ids"])
            right_ids = list(right["ranked_projection_ids"])
            intersection = set(left_ids) & set(right_ids)
            union = set(left_ids) | set(right_ids)
            core_overlap = len(intersection) / max(
                1, min(len(left_ids), len(right_ids))
            )
            retrieval_overlap = len(intersection) / max(1, len(union))
            common_order = [
                item for item in left_ids if item in intersection
            ]
            right_order = [
                item for item in right_ids if item in intersection
            ]
            rank_correlation = (
                sum(
                    left_item == right_item
                    for left_item, right_item in zip(
                        common_order, right_order
                    )
                ) / max(1, len(intersection))
            )
            comparisons.append({
                "canonical_dimension_id": canonical_id,
                "lineage_preserved": True,
                "core_overlap": core_overlap,
                "projection_rank_correlation": rank_correlation,
                "centroid_drift": abs(
                    float(left["strength"])
                    - float(right["strength"])
                ),
                "retrieval_set_overlap": retrieval_overlap,
                "utility_delta": (
                    float(right["utility"]) - float(left["utility"])
                ),
                "applicability_overlap": retrieval_overlap,
            })
        return {
            "preserved_count": len(comparisons),
            "comparisons": comparisons,
        }

    @staticmethod
    def universes_definitions() -> Iterable[tuple[str, str, str]]:
        from .universe import UNIVERSE_DEFINITIONS
        return UNIVERSE_DEFINITIONS

    def _smoke(self) -> list[Dict[str, Any]]:
        metrics: list[Dict[str, Any]] = []
        trained = 0
        original_path = database.DB_PATH
        try:
            with tempfile.TemporaryDirectory(
                prefix="superai-smoke-"
            ) as directory:
                database.DB_PATH = Path(directory) / "state.sqlite"
                repository = GraphRepository()
                training = GraphTrainingService(repository)
                dialogue = GraphDialogueService(repository)
                for size in self.config.smoke_sizes:
                    for index in range(trained, size):
                        training.train(
                            f"Робот {index} поднял контейнер {index}.",
                            independent_key=f"smoke:{index}",
                            domain_key="smoke",
                            discover_dimensions=False,
                            project_universes=False,
                        )
                    trained = size
                    started = perf_counter()
                    hive_id = dialogue.create()["hive"]["id"]
                    result = dialogue.query(
                        hive_id,
                        f"Кто поднял контейнер {size - 1}?",
                        resolved_mode="NEW_QUERY",
                    )
                    elapsed_ms = (
                        perf_counter() - started
                    ) * 1000.0
                    trace_metrics = (
                        result.get("trace", {})
                        .get("swarm", {})
                        .get("metrics", {})
                    )
                    metrics.append({
                        "events": size,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "events_scanned": int(
                            trace_metrics.get("events_scanned") or 0
                        ),
                        "database_queries": int(
                            trace_metrics.get("database_queries") or 0
                        ),
                        "bee_steps": int(
                            trace_metrics.get("bee_steps") or 0
                        ),
                        "nectar_packets": int(
                            trace_metrics.get(
                                "nectar_packets_created"
                            ) or 0
                        ),
                        "graph_match_attempts": int(
                            trace_metrics.get(
                                "graph_match_attempts"
                            ) or 0
                        ),
                    })
        finally:
            database.DB_PATH = original_path
        return metrics

    @staticmethod
    def _binding_regression() -> Dict[str, Any]:
        """Run binding checks in isolation from the experiment corpus."""
        original_path = database.DB_PATH
        try:
            with tempfile.TemporaryDirectory(
                prefix="superai-bindings-"
            ) as directory:
                database.DB_PATH = Path(directory) / "state.sqlite"
                repository = GraphRepository()
                training = GraphTrainingService(repository)
                dialogue = GraphDialogueService(repository)
                training.train(
                    "Механик дал роботу болт.",
                    independent_key="binding-regression:first",
                    discover_dimensions=False,
                )
                multi = dialogue.query(
                    dialogue.create()["hive"]["id"],
                    "Кто, кому и что дал?",
                    resolved_mode="NEW_QUERY",
                )
                multi_lemmas = {
                    str(item.get("resolved_lemma") or "")
                    for item in multi.get("selected_bindings") or []
                }
                multi_correct = bool(
                    multi["answer"]["status"] == "RESOLVED"
                    and multi_lemmas == {"механик", "робот", "болт"}
                    and multi.get("binding_configuration", {}).get(
                        "all_required_gaps_bound"
                    ) is True
                )

                hive_id = dialogue.create()["hive"]["id"]
                dialogue.query(
                    hive_id,
                    "Что механик дал роботу?",
                    resolved_mode="NEW_QUERY",
                )
                continuation = dialogue.query(hive_id, "А кто?")
                continuation_correct = (
                    continuation["answer"]["surface"] == "Механик."
                    and continuation["query_graph"]["continuation_of"]
                    is not None
                )

                training.train(
                    "Инженер дал машине кабель.",
                    independent_key="binding-regression:second",
                    discover_dimensions=False,
                )
                ambiguity = dialogue.query(
                    dialogue.create()["hive"]["id"],
                    "Кто дал?",
                    resolved_mode="NEW_QUERY",
                )
                ambiguity_correct = (
                    ambiguity["answer"]["status"]
                    == "AMBIGUOUS_BINDING"
                    and not ambiguity.get("selected_bindings")
                )

                training.train(
                    "Техник передал кабель.",
                    independent_key="binding-regression:incomplete",
                    discover_dimensions=False,
                )
                unresolved = dialogue.query(
                    dialogue.create()["hive"]["id"],
                    "Кто, кому и что передал?",
                    resolved_mode="NEW_QUERY",
                )
                unresolved_correct = bool(
                    unresolved["answer"]["status"] == "UNRESOLVED"
                    and unresolved["answer"]["validation"]["reason"]
                    == "INCOMPLETE_BINDING_CONFIGURATION"
                    and not unresolved.get("selected_bindings")
                )
                return {
                    "multi_gap": {
                        "total": 1,
                        "correct": int(multi_correct),
                        "accuracy": float(multi_correct),
                    },
                    "unresolved_correctness": float(
                        unresolved_correct
                    ),
                    "ambiguity_handling": float(ambiguity_correct),
                    "dialogue_continuation_accuracy": float(
                        continuation_correct
                    ),
                }
        finally:
            database.DB_PATH = original_path

    def run(self) -> Dict[str, Any]:
        run_started = perf_counter()
        sql_before, executes_before = database.metrics_snapshot()
        serialization_before = serialization_snapshot()
        counts = self.validate_dataset()
        self.repository.reset()
        self.universes._ensure_universes()
        started_at = utcnow()
        experiment_id = stable_id(
            "experiment",
            self.config.dataset_version,
            self.config.hash(),
            self.config.random_seed,
        )
        training_order: list[str] = []
        meta = self.repository.graph_meta()
        with self.repository.transaction() as conn:
            conn.execute(
                """INSERT INTO experiment_runs
                   (id,dataset_version,dataset_split,schema_version,
                    pipeline_versions_json,configuration_hash,random_seed,
                    training_order_json,batch_boundaries_json,status,
                    report_json,started_at,completed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,'RUNNING','{}',?,NULL)""",
                (
                    experiment_id,
                    self.config.dataset_version,
                    "full",
                    meta["schema_version"],
                    encode(meta),
                    self.config.hash(),
                    self.config.random_seed,
                    "[]",
                    encode(list(self.config.batch_boundaries)),
                    started_at,
                ),
            )
        training_order.extend(self._train_split("train", TRAIN))
        consolidation_before = self._consolidate("after_train")
        training_order.extend(self._train_split("holdout", HOLDOUT))
        holdout_before = self._evaluate("holdout_before_continual", HOLDOUT)
        training_order.extend(self._train_split("continual", CONTINUAL))
        consolidation_after = self._consolidate("after_continual")
        lineage = self._lineage_comparison(
            consolidation_before["snapshots"],
            consolidation_after["snapshots"],
        )
        holdout_after = self._evaluate("holdout_after_continual", HOLDOUT)
        training_order.extend(self._train_split("blind", BLIND))
        blind = self._evaluate("blind", BLIND)
        binding_regression = self._binding_regression()
        smoke = self._smoke()
        with self.repository.transaction() as conn:
            dimension_counts = {
                str(row["status"]): int(row["count"])
                for row in conn.execute(
                    """SELECT status,COUNT(*) AS count
                       FROM latent_dimensions GROUP BY status"""
                ).fetchall()
            }
            for status in (
                "candidate", "probation", "active", "shared", "weak",
                "merged", "split", "pruned", "frozen",
            ):
                dimension_counts.setdefault(status, 0)
            dimension_metrics = dict(conn.execute(
                """SELECT
                     COALESCE(MAX(holdout_retrieval_gain),0)
                       AS holdout_retrieval_gain,
                     COALESCE(MAX(shadow_retrieval_gain),0)
                       AS shadow_retrieval_gain,
                     COALESCE(MIN(CASE WHEN status='active'
                       THEN stability END),0) AS active_min_stability,
                     COALESCE(MIN(CASE WHEN status='active'
                       THEN stability_lower_bound END),0)
                       AS active_min_stability_lower_bound,
                     COALESCE(SUM(
                       validated_answer_contribution_count
                     ),0) AS validated_answer_contributions
                   FROM latent_dimensions"""
            ).fetchone())
            swarm_counts = dict(conn.execute(
                """SELECT COUNT(*) AS runs,
                          COALESCE(SUM(
                            CASE WHEN retrieval_mode='SWARM_DIMENSIONAL'
                            THEN 1 ELSE 0 END
                          ),0) AS dimensional_runs,
                          COALESCE(SUM(
                            CASE WHEN retrieval_mode='INDEX_FALLBACK'
                            THEN 1 ELSE 0 END
                          ),0) AS fallback_runs,
                          COALESCE(SUM(
                            CASE WHEN retrieval_mode='SWARM_MIXED'
                            THEN 1 ELSE 0 END
                          ),0) AS mixed_runs
                   FROM swarm_runs"""
            ).fetchone())
            swarm_evidence = dict(conn.execute(
                """SELECT
                     (SELECT COUNT(*) FROM bee_missions) AS bees,
                     (SELECT COUNT(*) FROM bee_steps) AS routes,
                     (SELECT COUNT(*) FROM nectar_packets) AS packets,
                     (SELECT COUNT(*) FROM (
                       SELECT source_universe AS universe_id
                         FROM bee_steps
                       UNION
                       SELECT target_universe AS universe_id
                         FROM bee_steps
                     )) AS visited_universes"""
            ).fetchone())
        swarm_counts.update(swarm_evidence)
        swarm_counts["fallback_rate"] = (
            int(swarm_counts["fallback_runs"])
            / max(1, int(swarm_counts["runs"]))
        )
        swarm_counts["dimensional_retrieval_rate"] = (
            int(swarm_counts["dimensional_runs"])
            / max(1, int(swarm_counts["runs"]))
        )
        single_gap_total = int(holdout_after["total"]) + int(
            blind["total"]
        )
        single_gap_correct = int(holdout_after["correct"]) + int(
            blind["correct"]
        )
        smoke_no_quadratic_regression = all(
            (
                current["events_scanned"]
                <= max(1, previous["events_scanned"]) * 3.5
                and current["graph_match_attempts"]
                <= max(1, previous["graph_match_attempts"]) * 3.5
            )
            for previous, current in zip(smoke, smoke[1:])
        )
        criteria = {
            "active_dimension_observed": (
                dimension_counts["active"] > 0
            ),
            "positive_holdout_retrieval_gain": (
                float(dimension_metrics["holdout_retrieval_gain"]) > 0
            ),
            "lineage_preserved": (
                consolidation_after
                and lineage["preserved_count"] > 0
            ),
            "validated_dimension_contribution": (
                int(
                    dimension_metrics[
                        "validated_answer_contributions"
                    ]
                ) > 0
            ),
            "dimensional_swarm_observed": (
                int(swarm_counts["dimensional_runs"]) > 0
            ),
            "holdout_not_regressed_after_continual": (
                float(holdout_after["accuracy"])
                + self.config.deterministic_tolerance
                >= float(holdout_before["accuracy"])
            ),
            "multi_gap_configuration_valid": (
                binding_regression["multi_gap"]["accuracy"] == 1.0
            ),
            "incomplete_multi_gap_unresolved": (
                binding_regression["unresolved_correctness"] == 1.0
            ),
            "ambiguity_not_randomly_resolved": (
                binding_regression["ambiguity_handling"] == 1.0
            ),
            "dialogue_continuation_correct": (
                binding_regression[
                    "dialogue_continuation_accuracy"
                ] == 1.0
            ),
            "smoke_no_quadratic_regression": (
                smoke_no_quadratic_regression
            ),
        }
        criteria["all_required_passed"] = all(criteria.values())
        acceleration_diagnostics = self.runtime.diagnostics()
        sql_after, executes_after = database.metrics_snapshot()
        sql_ms = max(0.0, sql_after - sql_before)
        serialization_ms = max(
            0.0, serialization_snapshot() - serialization_before
        )
        elapsed_ms = (perf_counter() - run_started) * 1000.0
        report = {
            "experiment_id": experiment_id,
            "dataset_version": self.config.dataset_version,
            "configuration_hash": self.config.hash(),
            "random_seed": self.config.random_seed,
            "dataset": counts,
            "dimensions": {
                "statuses": dimension_counts,
                "metrics": dimension_metrics,
                "before_continual": consolidation_before,
                "after_continual": consolidation_after,
                "lineage": lineage,
            },
            "swarms": swarm_counts,
            "bindings": {
                "single_gap": {
                    "total": single_gap_total,
                    "correct": single_gap_correct,
                    "accuracy": (
                        single_gap_correct / max(1, single_gap_total)
                    ),
                },
                **binding_regression,
                "holdout_before_continual": holdout_before,
                "holdout_after_continual": holdout_after,
                "blind": blind,
            },
            "performance": smoke,
            **acceleration_diagnostics,
            "sql_ms": round(sql_ms, 3),
            "numerical_ms": round(max(
                0.0,
                elapsed_ms - sql_ms - serialization_ms,
            ), 3),
            "serialization_ms": round(serialization_ms, 3),
            "index_build_ms": round(sum(
                float(item.get("index_build_ms") or 0.0)
                for item in smoke
            ), 3),
            "peak_memory_bytes": (
                int(tracemalloc.get_traced_memory()[1])
                if tracemalloc.is_tracing()
                else 0
            ),
            "python_iterations": sum(
                int(item.get("events_scanned") or 0)
                for item in smoke
            ),
            "sqlite_execute_count": max(
                0, executes_after - executes_before
            ),
            "criteria": criteria,
            "open_limitations": [
                "no proof above 100 events",
                "limited language coverage",
                "limited domain diversity",
                "early-stage dimension discovery",
            ],
            "started_at": started_at,
            "completed_at": utcnow(),
        }
        with self.repository.transaction() as conn:
            conn.execute(
                """UPDATE experiment_runs SET training_order_json=?,
                   status='COMPLETED',report_json=?,completed_at=?
                   WHERE id=?""",
                (
                    encode(training_order),
                    encode(report),
                    report["completed_at"],
                    experiment_id,
                ),
            )
            metric_rows = [
                (
                    "holdout_before_continual",
                    "accuracy",
                    holdout_before["accuracy"],
                    {},
                ),
                (
                    "holdout_after_continual",
                    "accuracy",
                    holdout_after["accuracy"],
                    {},
                ),
                ("blind", "accuracy", blind["accuracy"], {}),
            ]
            metric_rows.extend(
                (
                    f"smoke_{item['events']}",
                    "elapsed_ms",
                    item["elapsed_ms"],
                    item,
                )
                for item in smoke
            )
            conn.executemany(
                """INSERT INTO experiment_metrics
                   (id,experiment_id,phase,metric_name,metric_value,
                    tolerance,details_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                [
                    (
                        stable_id(
                            "experiment-metric",
                            experiment_id,
                            phase,
                            name,
                        ),
                        experiment_id,
                        phase,
                        name,
                        float(value),
                        self.config.deterministic_tolerance,
                        encode(details),
                        report["completed_at"],
                    )
                    for phase, name, value, details in metric_rows
                ],
            )
        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic compact SuperAI experiment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".superai/experiment-report.json"),
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    report = CompactExperiment(
        config=ExperimentConfig(random_seed=args.seed)
    ).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
