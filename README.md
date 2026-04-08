# 🚀 Humanitarian Aid Allocation — OpenEnv

> An LLM-driven reinforcement learning environment for optimising disaster relief logistics under uncertainty.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://www.docker.com/)
[![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-orange?logo=huggingface)](https://huggingface.co/docs/hub/spaces)

---

## Overview

**Meta\_Scalar** is a competition submission implementing a fully self-contained OpenEnv-compatible environment for humanitarian aid allocation. An LLM agent (default: `Qwen2.5-72B-Instruct`) acts as a logistics planner, deciding each turn how to distribute limited relief supplies across disaster-affected zones — balancing severity, population need, road conditions, and resource efficiency.

The environment is modelled as a **Markov Decision Process (MDP)** with stochastic demand, road blockages, supply shocks, and a multi-objective reward function that penalises both under-delivery and waste.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [MDP Specification](#mdp-specification)
- [Reward Function](#reward-function)
- [Grader / Score](#grader--score)
- [Task Configurations](#task-configurations)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

- **OpenEnv-compliant** REST API (`/reset`, `/step`, `/grade`, `/state`)
- **Three difficulty tiers** — easy, medium, hard — with configurable zones, steps, supply, and events
- **Stochastic dynamics** — new needs arrive each turn via Poisson processes; hard mode includes a mid-episode supply shock
- **Multi-objective reward** — weighted combination of coverage, population impact, waste penalty, critical-zone urgency, and equity (Gini coefficient)
- **LLM agent loop** — `inference.py` drives any OpenAI-compatible model via HuggingFace Inference Router
- **Typed schemas** — all state, action, and result objects are Pydantic v2 models
- **Dockerised** — production-ready `Dockerfile` for HuggingFace Spaces deployment

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                     inference.py                      │
│  ┌─────────────┐      ┌──────────────────────────┐   │
│  │  LLM Agent  │◄────►│  OpenAI-compatible client │   │
│  │  (Qwen 72B) │      │  (HuggingFace Router)     │   │
│  └──────┬──────┘      └──────────────────────────┘   │
│         │  JSON actions                                │
│  ┌──────▼──────────────────────────────────────────┐  │
│  │           FastAPI Server  (server/)             │  │
│  │   POST /reset  │  POST /step  │  POST /grade    │  │
│  └──────────────────────┬────────────────────────┘  │
│                          │                            │
│  ┌───────────────────────▼──────────────────────┐    │
│  │          HumanitarianAidEnv (MDP)            │    │
│  │  ZoneState × N  +  GlobalState  +  Grader    │    │
│  └──────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

---

## MDP Specification

| Component | Definition |
|-----------|-----------|
| **State S** | `(s₁, …, sₙ, g)` — per-zone states plus a global state |
| **Zone state sᵢ** | population `nᵢ`, severity `σᵢ ∈ [0,1]`, deficit `dᵢ`, coverage `cᵢ ∈ {0,1}`, road blockage flag |
| **Global state g** | remaining supply `R`, current step `t`, active event flags |
| **Action aₜ** | `zone_id`, `quantity q ≤ R`, `priority ∈ {low, med, high}` |
| **Transition — deficit** | `dᵢ(t+1) = max(0, dᵢ(t) − aᵢ(t)) + δᵢ` where `δᵢ ~ Poisson(λᵢ · σᵢ)` |
| **Transition — coverage** | `cᵢ(t+1) = 1` iff `dᵢ(t+1) = 0` |
| **Transition — supply** | `R(t+1) = R(t) − q` |
| **Waste** | `max(0, aᵢ − dᵢ)`, accumulated across the episode |

> **Road blockage rule:** Blocked zones only accept `priority="high"` (airdrop). Lower-priority deliveries fail silently without consuming supply.

---

## Reward Function

Each step returns a **dense reward** computed as:

```
r = w₁·Δcov_sev + w₂·Δpop_help − w₃·waste_frac − w₄·critical_unmet + w₅·Δgini_bonus
```

| Term | Weight | Description |
|------|--------|-------------|
| `Δcov_sev` | 0.40 | Sum of severity scores for newly-covered zones |
| `Δpop_help` | 0.30 | Population-weighted deficit reduction fraction |
| `waste_frac` | 0.15 | Fraction of sent supply that was wasted (over-delivery) |
| `critical_unmet` | 0.20 | Unmet need in critical zones (severity ≥ 0.8), normalised by population |
| `Δgini_bonus` | 0.10 | Reduction in Gini coefficient of coverage inequality |

---

## Grader / Score

The terminal grader produces a scalar `score ∈ (0, 1)`:

```
G = 0.6 · survival_rate + 0.3 · efficiency + 0.1 · time_bonus
```

| Sub-score | Description |
|-----------|-------------|
| `survival_rate` | Fraction of zones fully covered at episode end (clamped to `[0.05, 0.95]`) |
| `efficiency` | `1 − (total_waste / total_sent)` (clamped to `[0.05, 0.95]`) |
| `time_bonus` | `0.9` if all zones covered before final step, else `0.5` |

A score ≥ **0.5** is considered a **success**.

---

## Task Configurations

| Config | Zones | Steps | Initial Supply | Road Blocks | Supply Shock |
|--------|-------|-------|---------------|-------------|--------------|
| `easy` | 3 | 4 | 60 | None | No |
| `medium` | 5 | 6 | 100 | 2 zones | No |
| `hard` | 7 | 8 | 180 | 3 zones | Yes (at step 8, supply × Uniform(0.4, 0.6)) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Start the environment server

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run the inference agent

```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"   # or any OpenAI-compatible model
export TASK_NAME="easy"                           # easy | medium | hard

python inference.py
```

The agent will run all three difficulty tiers sequentially and print structured logs:

```
[START] task=easy env=humanitarian-aid-allocation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=zone=1 qty=18 prio=high reward=0.42 done=false error=null
[STEP] step=2 action=zone=2 qty=8 prio=med reward=0.31 done=false error=null
...
[END] success=true steps=4 score=0.812 rewards=0.42,0.31,...
```

---

## Running with Docker

```bash
# Build
docker build -t meta-scalar .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  meta-scalar
```

The server will be available at `http://localhost:7860`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` / `API_KEY` | — | API key for the inference provider |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Base URL for OpenAI-compatible LLM API |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | LLM model identifier |
| `ENV_BASE_URL` | `http://localhost:7860` | Base URL of the running environment server |
| `TASK_NAME` | `easy` | Default task when running a single episode |

---

## API Reference

All endpoints are served by the FastAPI server in `server/`.

### `POST /reset?task={easy|medium|hard}`
Resets the environment and returns the initial `Observation`.

### `POST /step`
Advances the environment by one step.

**Request body:**
```json
{ "zone_id": 2, "quantity": 15, "priority": "high" }
```

**Response:** `StepResult` containing `observation`, `reward`, `done`, and `info`.

### `POST /grade`
Returns the terminal `GraderResult` with `score`, `survival_rate`, `efficiency`, and `time_bonus`. Should be called after the episode is done.

### `GET /state`
Returns the full internal environment state (useful for debugging and checkpointing).

---

## Project Structure

```
Meta_Scalar/
├── humanitarian_env.py   # Core MDP environment (Pydantic models + HumanitarianAidEnv)
├── inference.py          # LLM agent inference loop
├── server/               # FastAPI server exposing the OpenEnv REST API
├── baseline_scores.json  # Reference scores for each difficulty tier
├── openenv.yaml          # OpenEnv specification metadata
├── Dockerfile            # Container build for HuggingFace Spaces
├── pyproject.toml        # Project metadata and tool config
├── requirements.txt      # Python dependencies
└── uv.lock               # Locked dependency tree
```

---

