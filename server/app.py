"""
server/app.py — Humanitarian Aid Allocation OpenEnv Server
FastAPI app exposing /reset, /step, /state, /grade endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from humanitarian_env import (
    Action,
    GraderResult,
    HumanitarianAidEnv,
    Observation,
    StepResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global env instance (single-session; fine for HF Spaces single-container)
# ---------------------------------------------------------------------------
_env: HumanitarianAidEnv | None = None


def get_env() -> HumanitarianAidEnv:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import datetime
    print(f"===== Application Startup at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} =====")
    yield


app = FastAPI(
    title="Humanitarian Aid Allocation OpenEnv",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset(task: str = "easy", seed: int | None = None):
    """Reset the environment and return the initial observation."""
    global _env
    _env = HumanitarianAidEnv(task=task, seed=seed)
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Apply an action and return the next observation, reward, and done flag."""
    env = get_env()
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def state():
    """Return the full internal environment state (for debugging)."""
    env = get_env()
    return JSONResponse(content=env.state())


@app.post("/grade", response_model=GraderResult)
def grade():
    """Return the terminal grader score. Call after episode ends."""
    env = get_env()
    return env.grade()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point  ← required by the validator
# ---------------------------------------------------------------------------

def main():
    """Start the Uvicorn server. Required entry point for openenv validate."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
