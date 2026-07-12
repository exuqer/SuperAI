"""FastAPI server for SuperAI"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from .training import get_training_manager
from .database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SuperAI API",
    description="Semantic space training with gravitational physics",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class TrainRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class TrainResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    words: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}
    time_ms: int = 0
    error: Optional[str] = None


class SpaceResponse(BaseModel):
    words: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}


class SessionCreateRequest(BaseModel):
    name: str = "Обучение"


class SessionResponse(BaseModel):
    id: str
    name: str
    created_at: float
    updated_at: float


class ResetResponse(BaseModel):
    success: bool
    words: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}
    error: Optional[str] = None


# API Endpoints
@app.post("/api/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """Train on text input."""
    manager = get_training_manager()
    if request.session_id:
        manager.set_session(request.session_id)
    result = manager.learn(request.text)
    return TrainResponse(**result)


# План гравитационного обучения использует более явный namespace. Оставляем
# короткие маршруты выше для обратной совместимости со старыми клиентами.
app.add_api_route("/api/v1/training/learn", train, methods=["POST"], response_model=TrainResponse)


@app.get("/api/space", response_model=SpaceResponse)
async def get_space(session_id: Optional[str] = Query(None)):
    """Get current word space state."""
    manager = get_training_manager()
    if session_id:
        manager.set_session(session_id)
    result = manager.get_space()
    return SpaceResponse(**result)


app.add_api_route("/api/v1/training/space", get_space, methods=["GET"], response_model=SpaceResponse)


@app.post("/api/reset", response_model=ResetResponse)
@app.delete("/api/v1/training/space", response_model=ResetResponse)
async def reset_space(session_id: Optional[str] = Query(None)):
    """Reset (clear) the word space."""
    manager = get_training_manager()
    if session_id:
        manager.set_session(session_id)
    result = manager.reset_space()
    return ResetResponse(**result)


app.add_api_route("/api/v1/training/reset", reset_space, methods=["POST"], response_model=ResetResponse)


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new training session."""
    from .database import create_session, get_session
    session_id = create_session(request.name)
    session = get_session(session_id)
    if not session:
        raise HTTPException(500, "Failed to create session")
    return SessionResponse(**session)


@app.get("/api/sessions", response_model=List[SessionResponse])
async def list_sessions():
    """List all training sessions."""
    from .database import list_sessions
    sessions = list_sessions()
    return [SessionResponse(**s) for s in sessions]


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session by ID."""
    from .database import get_session
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return SessionResponse(**session)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a training session."""
    from .database import get_session, delete_session
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    delete_session(session_id)
    return {"success": True}


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
