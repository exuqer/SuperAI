from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine import EngineConfig, SemanticEngine


@dataclass
class ServerConfig:
    state_dir: Path = Path(".semantic_ants")
    host: str = "127.0.0.1"
    port: int = 8000
    static_dir: Path | None = None


class EngineService:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.engine = SemanticEngine(
            config=EngineConfig(
                state_dir=config.state_dir,
            )
        )

    def health(self) -> dict[str, Any]:
        return self.engine.health()

    def app_config(self) -> dict[str, Any]:
        return self.engine.config_payload()

    def chat_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.engine.chat(
            str(payload.get("text", "")),
            session_id=str(payload.get("session_id") or "default"),
            backpack_limit=payload.get("backpack_limit"),
            include_graph=bool(payload.get("include_graph") or False),
            include_layers=bool(payload.get("include_layers") or False),
            include_trace=bool(payload.get("include_trace") or False),
        )

    def chat_backpack(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.engine.backpack(
            session_id=str(payload.get("session_id") or "default"),
            limit=payload.get("backpack_limit"),
            result_id=payload.get("result_id"),
            include_layers=bool(payload.get("include_layers") or False),
        )

    def chat_visuals(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.engine.chat_visuals(
            result_id=str(payload.get("result_id") or ""),
            session_id=str(payload.get("session_id") or "default"),
            backpack_limit=payload.get("backpack_limit"),
            graph_limit=payload.get("graph_limit"),
            include_layers=bool(payload.get("include_layers") or False),
        )

    def drill_down(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_id = str(payload.get("node_id") or "")
        if not node_id:
            raise KeyError("node_id")
        return self.engine.drill_down(
            node_id,
            session_id=str(payload.get("session_id") or "default"),
            limit=payload.get("limit"),
        )

    def drill_up(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.engine.drill_up(
            session_id=str(payload.get("session_id") or "default"),
            limit=payload.get("limit"),
        )

    def submit_train(self, payload: dict[str, Any]):
        dataset_path = payload.get("dataset_path")
        if dataset_path:
            return self.engine.jobs.submit(
                "train",
                self.engine.train_jsonl,
                path=str(dataset_path),
                session_id=str(payload.get("session_id") or "default"),
                epochs=int(payload.get("epochs") or 1),
                max_pairs=payload.get("max_pairs"),
            )
        return self.engine.jobs.submit(
            "train",
            self.engine.train_job,
            text=str(payload.get("text", "")),
            session_id=str(payload.get("session_id") or "default"),
            epochs=int(payload.get("epochs") or 1),
        )

    def feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.engine.feedback(
            result_id=str(payload.get("result_id") or ""),
            score=int(payload.get("score") or 0),
            corrected_response=payload.get("corrected_response"),
        )

    def graph(self, filters: dict[str, Any]) -> dict[str, Any]:
        return self.engine.graph(
            query=filters.get("query"),
            limit=int(filters.get("limit") or self.engine.config.graph_limit),
            result_id=filters.get("result_id"),
        )

    def node_detail(self, node_id: str) -> dict[str, Any]:
        return self.engine.node_detail(node_id)

    def sessions(self) -> list[dict[str, Any]]:
        return self.engine.sessions()

    def reset_session(self, session_id: str) -> dict[str, Any]:
        return self.engine.reset_session(session_id)

    def jobs(self) -> list[dict[str, Any]]:
        return [job.to_dict() for job in self.engine.jobs.list()]

    def job(self, job_id: str) -> dict[str, Any] | None:
        job = self.engine.jobs.get(job_id)
        return job.to_dict() if job else None

    def shutdown(self) -> None:
        self.engine.flush_pending_persist()
        self.engine.jobs.shutdown()
