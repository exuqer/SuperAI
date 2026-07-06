from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without web extra
    raise RuntimeError('Install web dependencies with: pip install -e ".[web]"') from exc

from semantic_ants.server.schemas import (
    AnalyzeRequest,
    BootstrapRequest,
    DecodeRequest,
    DreamRequest,
    EvalRequest,
    ExportRequest,
    FeedbackRequest,
    KozievDownloadRequest,
    JsonlJobRequest,
    ResetNetworkRequest,
    SimpleTrainingRequest,
    SpcDownloadRequest,
    TatoebaDownloadRequest,
    UnderstandRequest,
    VectorInterpretRequest,
    model_payload,
)
from semantic_ants.server.service import EngineService, ServerConfig


def create_app(config: ServerConfig | None = None) -> FastAPI:
    runtime = config or ServerConfig()
    service = EngineService(runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            service.jobs.shutdown()

    app = FastAPI(title="semantic_ants web API", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            f"http://{runtime.host}:{runtime.port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/config")
    def app_config() -> dict[str, Any]:
        return service.app_config()

    @app.post("/api/analyze")
    def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
        return service.analyze(model_payload(payload))

    @app.post("/api/understand")
    def understand(payload: UnderstandRequest) -> dict[str, Any]:
        return service.understand(model_payload(payload))

    @app.post("/api/decode")
    def decode(payload: DecodeRequest) -> dict[str, Any]:
        return service.decode(model_payload(payload))

    @app.post("/api/chat/message")
    def chat_message(payload: AnalyzeRequest) -> dict[str, Any]:
        return service.chat_message(model_payload(payload))

    @app.get("/api/chat/sessions")
    def chat_sessions() -> list[dict[str, Any]]:
        return service.sessions()

    @app.delete("/api/chat/sessions/{session_id}")
    def reset_chat_session(session_id: str) -> dict[str, Any]:
        return service.reset_session(session_id)

    @app.post("/api/feedback")
    def feedback(payload: FeedbackRequest) -> dict[str, Any]:
        return service.feedback(model_payload(payload))

    @app.post("/api/vector/interpret")
    def interpret_vector(payload: VectorInterpretRequest) -> dict[str, Any]:
        return service.interpret_vector(model_payload(payload))

    @app.get("/api/memory/summary")
    def memory_summary() -> dict[str, Any]:
        return service.memory_summary()

    @app.get("/api/memory/results")
    def memory_results() -> list[dict[str, Any]]:
        return service.results()

    @app.get("/api/memory/collections")
    def memory_collections() -> dict[str, Any]:
        return service.memory_collections()

    @app.get("/api/concepts")
    def concepts(
        query: Optional[str] = None,
        layer: Optional[int] = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        return service.concepts(query=query, layer=layer, limit=limit)

    @app.get("/api/concepts/detail")
    def concept_detail(uri: str, result_id: Optional[str] = None) -> dict[str, Any]:
        return service.concept_detail(uri=uri, result_id=result_id)

    @app.get("/api/graph")
    def graph(
        layer: Optional[int] = None,
        source: Optional[str] = None,
        edge_type: Optional[str] = None,
        relation: Optional[str] = None,
        query: Optional[str] = None,
        min_pheromone: Optional[float] = None,
        only_signal: bool = False,
        result_id: Optional[str] = None,
        limit: Optional[int] = Query(default=800, ge=1, le=5000),
    ) -> dict[str, Any]:
        return service.graph(
            {
                "layer": layer,
                "source": source,
                "edge_type": edge_type,
                "relation": relation,
                "query": query,
                "min_pheromone": min_pheromone,
                "only_signal": only_signal,
                "result_id": result_id,
                "limit": limit,
            }
        )

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return service.job_list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        item = service.job(job_id)
        if not item:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post("/api/training/train")
    def train(payload: JsonlJobRequest) -> dict[str, Any]:
        return service.submit_train(model_payload(payload)).to_dict()

    @app.post("/api/training/learn")
    def learn(payload: JsonlJobRequest) -> dict[str, Any]:
        return service.submit_learn(model_payload(payload)).to_dict()

    @app.post("/api/training/learn-dialogues")
    def learn_dialogues(payload: JsonlJobRequest) -> dict[str, Any]:
        return service.submit_learn_dialogues(model_payload(payload)).to_dict()

    @app.post("/api/training/simple")
    def simple_train(payload: SimpleTrainingRequest) -> dict[str, Any]:
        return service.submit_simple_train(model_payload(payload)).to_dict()

    @app.post("/api/eval")
    def evaluate(payload: EvalRequest) -> dict[str, Any]:
        return service.submit_eval(model_payload(payload)).to_dict()

    @app.post("/api/system/dream")
    def dream(payload: DreamRequest) -> dict[str, Any]:
        return service.submit_dream(model_payload(payload)).to_dict()

    @app.post("/api/system/bootstrap")
    def bootstrap(payload: BootstrapRequest) -> dict[str, Any]:
        return service.submit_bootstrap(model_payload(payload)).to_dict()

    @app.post("/api/system/reset-network")
    def reset_network(payload: ResetNetworkRequest) -> dict[str, Any]:
        return service.submit_reset_network(model_payload(payload)).to_dict()

    @app.post("/api/datasets/spc/download")
    def download_spc(payload: SpcDownloadRequest) -> dict[str, Any]:
        return service.submit_download_spc(model_payload(payload)).to_dict()

    @app.post("/api/datasets/koziev/download")
    def download_koziev(payload: KozievDownloadRequest) -> dict[str, Any]:
        return service.submit_download_koziev(model_payload(payload)).to_dict()

    @app.post("/api/datasets/tatoeba/download")
    def download_tatoeba(payload: TatoebaDownloadRequest) -> dict[str, Any]:
        return service.submit_download_tatoeba(model_payload(payload)).to_dict()

    @app.post("/api/system/export")
    def export(payload: ExportRequest) -> dict[str, Any]:
        return service.submit_export(model_payload(payload)).to_dict()

    static_dir = _static_dir(runtime.static_dir)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")

    return app


def _static_dir(path: Path | None) -> Path | None:
    if path is not None:
        return path if path.exists() else None
    candidate = Path(__file__).resolve().parents[2] / "web" / "dist"
    return candidate if candidate.exists() else None
