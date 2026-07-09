from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from .schemas import ChatRequest, DrillRequest, FeedbackRequest, TrainRequest, model_payload
from .service import EngineService, ServerConfig


def create_app(config: ServerConfig | None = None) -> FastAPI:
    runtime = config or ServerConfig()
    service = EngineService(runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="semantic_ants", version="0.2.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/config")
    def app_config() -> dict[str, Any]:
        return service.app_config()

    @app.post("/api/chat/message")
    def chat_message(payload: ChatRequest) -> dict[str, Any]:
        return service.chat_message(model_payload(payload))

    @app.get("/api/chat/backpack")
    def chat_backpack(
        session_id: str = "default",
        result_id: str | None = None,
        backpack_limit: int | None = None,
        layers: bool = False,
    ) -> dict[str, Any]:
        try:
            return service.chat_backpack(
                {
                    "session_id": session_id,
                    "result_id": result_id,
                    "backpack_limit": backpack_limit,
                    "include_layers": layers,
                }
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="result not found") from exc

    @app.get("/api/chat/results/{result_id}/visuals")
    def chat_visuals(
        result_id: str,
        session_id: str = "default",
        backpack_limit: int | None = None,
        graph_limit: int | None = None,
        layers: bool = False,
    ) -> dict[str, Any]:
        try:
            return service.chat_visuals(
                {
                    "result_id": result_id,
                    "session_id": session_id,
                    "backpack_limit": backpack_limit,
                    "graph_limit": graph_limit,
                    "include_layers": layers,
                }
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="result not found") from exc

    @app.post("/api/chat/drill-down")
    def drill_down(payload: DrillRequest) -> dict[str, Any]:
        try:
            return service.drill_down(model_payload(payload))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="node not found") from exc

    @app.post("/api/chat/drill-up")
    def drill_up(payload: DrillRequest) -> dict[str, Any]:
        return service.drill_up(model_payload(payload))

    @app.get("/api/chat/sessions")
    def chat_sessions() -> list[dict[str, Any]]:
        return service.sessions()

    @app.delete("/api/chat/sessions/{session_id}")
    def reset_chat_session(session_id: str) -> dict[str, Any]:
        return service.reset_session(session_id)

    @app.post("/api/train")
    def train(payload: TrainRequest) -> dict[str, Any]:
        return service.submit_train(model_payload(payload)).to_dict()

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return service.jobs()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        item = service.job(job_id)
        if not item:
            raise HTTPException(status_code=404, detail="job not found")
        return item

    @app.post("/api/feedback")
    def feedback(payload: FeedbackRequest) -> dict[str, Any]:
        return service.feedback(model_payload(payload))

    @app.get("/api/graph")
    def graph(
        query: str | None = None,
        limit: int = Query(default=120, ge=1, le=1000),
        result_id: str | None = None,
    ) -> dict[str, Any]:
        return service.graph({"query": query, "limit": limit, "result_id": result_id})

    @app.get("/api/node/{node_id}")
    def node_detail(node_id: str) -> dict[str, Any]:
        try:
            return service.node_detail(node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="node not found") from exc

    static_dir = _static_dir(runtime.static_dir)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")

    return app


def _static_dir(path: Path | None) -> Path | None:
    if path is not None:
        return path if path.exists() else None
    candidate = Path(__file__).resolve().parents[2] / "web"
    return candidate if candidate.exists() else None
