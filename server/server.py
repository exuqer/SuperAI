"""FastAPI server for the relation-free concept field."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .database import init_db
from .training import get_training_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SuperAI API",
    description="Relation-free semantic concept field",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TrainResponse(BaseModel):
    success: bool
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    time_ms: int = 0
    error: Optional[str] = None


class SpaceResponse(BaseModel):
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    success: bool
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


async def train(request: TrainRequest) -> TrainResponse:
    return TrainResponse(**get_training_manager().learn(request.text))


async def get_space() -> SpaceResponse:
    return SpaceResponse(**get_training_manager().get_space())


async def reset() -> ResetResponse:
    return ResetResponse(**get_training_manager().reset_space())


app.post("/api/train", response_model=TrainResponse)(train)
app.post("/api/v1/training/learn", response_model=TrainResponse)(train)
app.get("/api/space", response_model=SpaceResponse)(get_space)
app.get("/api/v1/training/space", response_model=SpaceResponse)(get_space)
app.post("/api/reset", response_model=ResetResponse)(reset)
app.delete("/api/v1/training/space", response_model=ResetResponse)(reset)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
