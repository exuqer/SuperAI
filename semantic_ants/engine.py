from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from semantic_ants.ants import AntColony, AntConfig
from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import ConceptNode, SemanticEdge, SemanticResult
from semantic_ants.core.normalization import detect_language, text_to_concept_uri, tokenize
from semantic_ants.generation import Interpreter, SemanticVectorInterpreter, TorchDialogueNavigator
from semantic_ants.knowledge import bootstrap_builtin_knowledge
from semantic_ants.knowledge.isolation import ensure_isolated_concept_edges
from semantic_ants.learning.aco import Judge, SemanticThought
from semantic_ants.learning.checkpoint import Checkpoint, CheckpointStore, default_checkpoint_path
from semantic_ants.providers import ConceptNetClient, ConceptNetError, JsonCache
from semantic_ants.understanding import understand_text


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
    strength_vector: tuple[int, ...] = ()


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
        self.store = store or CheckpointStore(default_checkpoint_path(self.config.state_dir))
        self.checkpoint = checkpoint or self.store.load()
        if self.config.autoload_builtin:
            report = bootstrap_builtin_knowledge(self.checkpoint, allow_network=self.config.allow_network)
            if report.changed:
                self.store.save(self.checkpoint)
        self.model_dir = self.config.state_dir / "models"
        self.speech = TorchDialogueNavigator()
        self.interpreter = Interpreter(navigator=self.speech, model_dir=self.model_dir)
        self.vector_interpreter = SemanticVectorInterpreter(navigator=self.speech, model_dir=self.model_dir)
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
        session_id: str | None = None,
        reset_session: bool = False,
        generate_response: bool = True,
        strength_vector: tuple[int, ...] | list[int] | None = None,
    ) -> SemanticResult:
        result, _ = self.analyze_with_graph(
            text=text,
            lang=lang,
            persist_result=persist_result,
            ant_count=ant_count,
            max_depth=max_depth,
            top_concepts=top_concepts,
            mode=mode,
            candidates=candidates,
            session_id=session_id,
            reset_session=reset_session,
            generate_response=generate_response,
            strength_vector=strength_vector,
        )
        return result

    def analyze_with_graph(
        self,
        text: str,
        lang: str = "auto",
        persist_result: bool = True,
        ant_count: int | None = None,
        max_depth: int | None = None,
        top_concepts: int | None = None,
        mode: str = "graph",
        candidates: int = 3,
        session_id: str | None = None,
        reset_session: bool = False,
        generate_response: bool = True,
        strength_vector: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[SemanticResult, SemanticGraph]:
        if reset_session:
            self.checkpoint.reset_chat_session(session_id)
        chat_history = self.checkpoint.session_history(
            session_id,
            limit=self.speech.config.max_context_turns,
        )
        selected_lang = self._select_lang(text, lang)
        tokens = tokenize(text)
        graph = SemanticGraph()
        start_uris = self._seed_graph(tokens, selected_lang, graph)
        graph.add_edges(self.checkpoint.learned_edges(), include_reverse=True)
        self._connect_input_concepts_to_top(text, selected_lang, start_uris, graph)
        selected_strength = tuple(strength_vector) if strength_vector is not None else self.config.strength_vector
        routes = AntColony(
            AntConfig(
                ant_count=ant_count or self.config.ant_count,
                max_depth=max_depth or self.config.max_depth,
                seed=self.checkpoint.seed or self.config.seed,
                strength_vector=selected_strength,
            )
        ).search(graph, start_uris, self.checkpoint, context_key=f"{selected_lang}:{text}")
        activated, summary, response, semantic_vector = self.interpreter.interpret(
            input_text=text,
            tokens=tokens,
            routes=routes,
            graph=graph,
            checkpoint=self.checkpoint,
            top_concepts=top_concepts or self.config.top_concepts,
            chat_history=chat_history,
            generate_response=generate_response,
            strength_vector=selected_strength,
            lang=selected_lang,
        )
        sources = sorted({edge.source for edge in graph.edges()})
        signal_trace = self._signal_trace(routes)
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
            session_id=session_id,
            context_turns=chat_history,
            semantic_vector=semantic_vector,
            signal_trace=signal_trace,
            response_source=str(semantic_vector.get("response_source", "")),
            response_lang=str(semantic_vector.get("response_lang", selected_lang)),
            response_candidates=[str(value) for value in semantic_vector.get("response_candidates", []) if value],
            canonical_concepts=[str(value) for value in self.checkpoint.canonical_concepts.keys()],
        )
        if mode == "hybrid" and generate_response:
            result = self._hybridize(result, candidates=candidates)
        if persist_result:
            self.checkpoint.remember_result(result.to_dict())
            concepts = [str(item.get("uri")) for item in result.activated_concepts if item.get("uri")]
            self.checkpoint.remember_chat_turn(session_id, "user", text, result.result_id, concepts)
            self.checkpoint.remember_chat_turn(session_id, "assistant", result.response, result.result_id, concepts)
            self.store.save(self.checkpoint)
        return result, graph

    def interpret_vector(self, semantic_vector: dict[str, object] | list[dict[str, object]]) -> str:
        return self.vector_interpreter.interpret(semantic_vector, self.checkpoint)

    def _hybridize(self, result: SemanticResult, candidates: int = 3) -> SemanticResult:
        return result

    def _select_lang(self, text: str, lang: str) -> str:
        requested = self.config.lang if lang == "auto" else lang
        return detect_language(text) if requested == "auto" else requested

    def _seed_graph(self, tokens: list[str], lang: str, graph: SemanticGraph) -> list[str]:
        start_uris: list[str] = []
        for token in tokens:
            raw_uri = self.checkpoint.aliases.get(token) or text_to_concept_uri(token, lang)
            uri = self.checkpoint.canonical_uri(raw_uri)
            self.checkpoint.register_canonical_concept(
                uri,
                aliases=[token],
                lang=lang,
                source_uri=raw_uri,
            )
            self.checkpoint.register_surface_form(uri, token, lang=lang)
            start_uris.append(uri)
            graph.add_node(ConceptNode(uri=uri, label=token.replace("_", " "), language=lang, source="input"))
            try:
                nodes, edges = self.client.edges_for(raw_uri, limit=self.config.conceptnet_limit)
            except ConceptNetError:
                continue
            for node in nodes:
                canonical_node_uri = self.checkpoint.canonical_uri(node.uri)
                graph.add_node(
                    ConceptNode(
                        uri=canonical_node_uri,
                        label=node.label,
                        language=node.language,
                        source=node.source,
                        layer=node.layer,
                        metadata={**node.metadata, "source_uri": node.uri},
                    )
                )
            graph.add_edges([self._canonical_edge(edge) for edge in edges], include_reverse=True)
        self._connect_input_tokens(start_uris, graph)
        return start_uris

    def _canonical_edge(self, edge: SemanticEdge) -> SemanticEdge:
        return SemanticEdge(
            start=self.checkpoint.canonical_uri(edge.start),
            end=self.checkpoint.canonical_uri(edge.end),
            relation=edge.relation,
            weight=edge.weight,
            source=edge.source,
            surface_text=edge.surface_text,
            layer=edge.layer,
            distance=edge.distance,
            edge_type=edge.edge_type,
            metadata={**edge.metadata, "source_start": edge.start, "source_end": edge.end},
        )

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

    def _connect_input_concepts_to_top(
        self,
        text: str,
        lang: str,
        start_uris: list[str],
        graph: SemanticGraph,
    ) -> None:
        understood = understand_text(text, lang=lang, checkpoint=self.checkpoint)
        content_uris = {
            self.checkpoint.canonical_uri(str(token.concept_uri))
            for token in understood.tokens
            if token.concept_uri and not token.is_stop_word
        }
        top_domains = {
            self.checkpoint.canonical_uri(str(token.concept_uri)): self._inferred_top_domain(token)
            for token in understood.tokens
            if token.concept_uri and not token.is_stop_word
        }
        for uri in start_uris:
            if uri in content_uris:
                ensure_isolated_concept_edges(graph, uri, lang=lang, top_domain=top_domains.get(uri))

    def _inferred_top_domain(self, token: object) -> str:
        morphology = getattr(token, "morphology", {}) or {}
        pos = str(morphology.get("POS") or "")
        if pos in {"VERB", "INFN", "PRTS", "GRND"}:
            return "/m/top/action"
        if pos in {"NPRO"}:
            return "/m/top/person"
        return "/m/top/object"

    def _result_id(self, text: str) -> str:
        base = f"{time.time_ns()}:{text}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

    def _signal_trace(self, routes: list[object]) -> list[dict[str, object]]:
        trace: list[dict[str, object]] = []
        for route in routes[:8]:
            for index, step in enumerate(route.steps):
                trace.append(
                    {
                        "ant_id": route.ant_id,
                        "step_index": index,
                        "start": step.start,
                        "end": step.end,
                        "relation": step.relation,
                        "layer": step.layer,
                        "distance": step.distance,
                        "remaining_strength": step.remaining_strength,
                        "edge_type": step.edge_type,
                        "score": round(float(step.score), 4),
                    }
                )
        return trace
