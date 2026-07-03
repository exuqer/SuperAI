from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path

from semantic_ants.ants import AntColony, AntConfig
from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import ConceptNode, SemanticEdge, SemanticResult
from semantic_ants.core.normalization import detect_language, text_to_concept_uri, tokenize
from semantic_ants.generation import Interpreter, MiniTransformerSpeechModule
from semantic_ants.knowledge import bootstrap_builtin_knowledge
from semantic_ants.learning.aco import Judge, SemanticThought
from semantic_ants.learning.checkpoint import Checkpoint, CheckpointStore
from semantic_ants.providers import ConceptNetClient, ConceptNetError, JsonCache


@dataclass(frozen=True)
class EngineConfig:
    state_dir: Path = Path(".semantic_ants")
    lang: str = "auto"
    ant_count: int = 32
    max_depth: int = 4
    top_concepts: int = 5
    conceptnet_limit: int = 30
    allow_network: bool = True
    seed: int = 42
    autoload_builtin: bool = True


class SemanticEngine:
    """Главный фасад анализа: слова -> концепты -> маршруты -> ответ."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        client: ConceptNetClient | None = None,
        store: CheckpointStore | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        cache = JsonCache(self.config.state_dir / "cache")
        self.client = client or ConceptNetClient(cache=cache, allow_network=self.config.allow_network)
        self.store = store or CheckpointStore(self.config.state_dir / "checkpoints" / "model.json")
        self.checkpoint = checkpoint or self.store.load()
        if self.config.autoload_builtin:
            report = bootstrap_builtin_knowledge(self.checkpoint)
            if report.changed:
                self.store.save(self.checkpoint)
        self.interpreter = Interpreter()
        self.speech = MiniTransformerSpeechModule()
        self.judge = Judge()

    def analyze(
        self,
        text: str,
        lang: str = "auto",
        persist_result: bool = True,
        ant_count: int | None = None,
        max_depth: int | None = None,
        top_concepts: int | None = None,
        mode: str = "graph",
        candidates: int = 3,
    ) -> SemanticResult:
        selected_lang = self._select_lang(text, lang)
        tokens = tokenize(text)
        graph = SemanticGraph()
        start_uris = self._seed_graph(tokens, selected_lang, graph)
        graph.add_edges(self.checkpoint.learned_edges(), include_reverse=True)
        routes = AntColony(
            AntConfig(
                ant_count=ant_count or self.config.ant_count,
                max_depth=max_depth or self.config.max_depth,
                seed=self.checkpoint.seed or self.config.seed,
            )
        ).search(graph, start_uris, self.checkpoint, context_key=f"{selected_lang}:{text}")
        activated, summary, response = self.interpreter.interpret(
            input_text=text,
            tokens=tokens,
            routes=routes,
            graph=graph,
            checkpoint=self.checkpoint,
            top_concepts=top_concepts or self.config.top_concepts,
        )
        sources = sorted({edge.source for edge in graph.edges()})
        result = SemanticResult(
            result_id=self._result_id(text),
            input_text=text,
            lang=selected_lang,
            tokens=tokens,
            activated_concepts=activated,
            routes=routes,
            summary=summary,
            response=response,
            sources=sources,
        )
        if mode == "hybrid":
            result = self._hybridize(result, candidates=candidates)
        if persist_result:
            self.checkpoint.remember_result(result.to_dict())
            self.store.save(self.checkpoint)
        return result

    def _hybridize(self, result: SemanticResult, candidates: int = 3) -> SemanticResult:
        thought = SemanticThought.from_result(result)
        generated = self.speech.generate(
            thought.to_prompt(),
            self.checkpoint,
            fallback=result.response,
            count=candidates,
        )
        answer, _ = self.judge.rank_freeform(result.input_text, thought, generated)
        return replace(result, response=answer)

    def _select_lang(self, text: str, lang: str) -> str:
        requested = self.config.lang if lang == "auto" else lang
        return detect_language(text) if requested == "auto" else requested

    def _seed_graph(self, tokens: list[str], lang: str, graph: SemanticGraph) -> list[str]:
        start_uris: list[str] = []
        for token in tokens:
            uri = self.checkpoint.aliases.get(token) or text_to_concept_uri(token, lang)
            start_uris.append(uri)
            graph.add_node(ConceptNode(uri=uri, label=token.replace("_", " "), language=lang, source="input"))
            try:
                nodes, edges = self.client.edges_for(uri, limit=self.config.conceptnet_limit)
            except ConceptNetError:
                nodes, edges = self._fallback_edges(uri, token, lang)
            for node in nodes:
                graph.add_node(node)
            graph.add_edges(edges, include_reverse=True)
        self._connect_input_tokens(start_uris, graph)
        return start_uris

    def _fallback_edges(
        self,
        uri: str,
        token: str,
        lang: str,
    ) -> tuple[list[ConceptNode], list[SemanticEdge]]:
        root_uri = f"/c/{lang}/unknown_context"
        node = ConceptNode(uri=root_uri, label="unknown context", language=lang, source="fallback")
        edge = SemanticEdge(
            start=uri,
            end=root_uri,
            relation="RelatedTo",
            weight=0.1,
            source="fallback",
            surface_text=f"Fallback-связь для {token}",
        )
        return [node], [edge]

    def _connect_input_tokens(self, start_uris: list[str], graph: SemanticGraph) -> None:
        for left, right in zip(start_uris, start_uris[1:]):
            graph.add_edge(
                SemanticEdge(
                    start=left,
                    end=right,
                    relation="ContextNeighbor",
                    weight=0.6,
                    source="input",
                ),
                include_reverse=True,
            )

    def _result_id(self, text: str) -> str:
        base = f"{time.time_ns()}:{text}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
