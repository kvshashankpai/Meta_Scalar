"""
inference.py — Humanitarian Aid Allocation Inference Script
============================================================
Mandatory stdout format:
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — all from environment variables
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASK_NAME   = os.getenv("TASK_NAME", "easy")
BENCHMARK   = "humanitarian-aid-allocation"
MAX_STEPS   = 8
TEMPERATURE = 0.3
MAX_TOKENS  = 256

SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert humanitarian aid logistics agent.
    You will receive the current state of disaster-affected zones and must decide
    how to allocate limited relief supplies.

    Each zone has: zone_id, population, severity (0-1), deficit (units needed), 
    covered (bool), road_blocked (bool).
    Global state: remaining_supply, current_step, total_steps.

    Rules:
    - Road-blocked zones require priority="high" (airdrop); other priorities fail silently.
    - Sending more than the deficit wastes supply.
    - Prioritise high-severity, high-population, uncovered zones.

    Respond ONLY with a valid JSON object, nothing else. Example:
    {"zone_id": 2, "quantity": 15, "priority": "high"}
""").strip()


def build_user_prompt(obs: dict) -> str:
    zones_text = json.dumps(obs["zones"], indent=2)
    g = obs["global_state"]
    return (
        f"Remaining supply: {g['remaining_supply']} | "
        f"Step {g['current_step']}/{g['total_steps']}\n\n"
        f"Zone states:\n{zones_text}\n\n"
        f"Hint: {obs.get('feasible_actions_hint', '')}\n\n"
        "Respond with JSON: {\"zone_id\": <int>, \"quantity\": <int>, \"priority\": \"low\"|\"med\"|\"high\"}"
    )


def get_action(client: OpenAI, obs: dict) -> dict:
    """Call the LLM and parse a valid action JSON."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Fallback: allocate to first uncovered zone
        for z in obs["zones"]:
            if not z["covered"] and z["deficit"] > 0:
                prio = "high" if z["road_blocked"] else "med"
                qty = min(z["deficit"], obs["global_state"]["remaining_supply"])
                return {"zone_id": z["zone_id"], "quantity": qty, "priority": prio}
        return {"zone_id": 0, "quantity": 0, "priority": "med"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_episode(task: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=30.0) as http:

            # Reset
            r = await http.post("/reset", params={"task": task})
            r.raise_for_status()
            obs = r.json()

            for step in range(1, MAX_STEPS + 1):
                action_dict = get_action(client, obs)
                action_str = (
                    f"zone={action_dict.get('zone_id')} "
                    f"qty={action_dict.get('quantity')} "
                    f"prio={action_dict.get('priority', 'med')}"
                )

                sr = await http.post("/step", json=action_dict)
                sr.raise_for_status()
                result = sr.json()

                reward = result["reward"]["value"]
                done   = result["done"]
                error  = result.get("info", {}).get("error")

                rewards.append(reward)
                steps_taken = step
                obs = result["observation"]

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

            # Grade
            gr = await http.post("/grade")
            gr.raise_for_status()
            grade_result = gr.json()
            score   = grade_result["score"]
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        await run_episode(task)


if __name__ == "__main__":
    asyncio.run(main())
