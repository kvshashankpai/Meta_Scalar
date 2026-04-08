"""
humanitarian_env.py — Humanitarian Aid Allocation OpenEnv Environment

MDP:
  State S = (s_1, ..., s_n, g)
    Zone state s_i: population n_i, severity σ_i ∈ [0,1], deficit d_i, coverage c_i ∈ {0,1}
    Global state g: remaining supply R, step t, road blockages B ∈ {0,1}^N, events E

  Action a_t: zone_id, quantity q ≤ R, priority ∈ {low, med, high}

  Transitions:
    d_i(t+1) = max(0, d_i(t) - a_i(t)) + δ_i   where δ_i ~ Poisson(λ_i · σ_i)
    c_i(t+1) = 1 iff d_i(t+1) = 0
    R(t+1) = R(t) - q
    waste = max(0, a_i - d_i) accumulated

  Reward (dense):
    r = w1·Δcov_sev + w2·Δpop_help - w3·waste_frac - w4·critical_unmet + w5·Δgini_bonus

  Terminal grader score G ∈ [0,1]:
    G = 0.6·survival_rate + 0.3·efficiency + 0.1·time_bonus
"""

from __future__ import annotations

import random
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models — OpenEnv typed spec
# ---------------------------------------------------------------------------

class ZoneState(BaseModel):
    zone_id: int
    population: int
    severity: float = Field(ge=0.0, le=1.0)
    deficit: int = Field(ge=0)
    covered: bool
    road_blocked: bool
    lambda_rate: float  # Poisson arrival rate for new needs


class GlobalState(BaseModel):
    remaining_supply: int
    current_step: int
    total_steps: int
    active_event_flags: int = 0
    supply_shock_applied: bool = False


class Observation(BaseModel):
    zones: List[ZoneState]
    global_state: GlobalState
    feasible_actions_hint: str = ""


class Action(BaseModel):
    zone_id: int = Field(ge=0, description="Target zone index (0-based)")
    quantity: int = Field(ge=0, description="Units of supply to send (0 = skip/noop)")
    priority: str = Field(default="med", description="Transport tier: low | med | high")


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    survival_rate: float
    efficiency: float
    time_bonus: float
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "easy": {
        "n_zones": 3,
        "total_steps": 4,
        "initial_supply": 60,
        "supply_shock": False,
        "populations":   [500, 800, 300],
        "severities":    [0.3, 0.6, 0.8],
        "initial_defs":  [10,  18,  8  ],
        "road_blocked":  [False, False, False],
        "lambda_rates":  [0.1, 0.2, 0.1],  # Greatly reduced new need arrivals
    },
    "medium": {
        "n_zones": 5,
        "total_steps": 6,
        "initial_supply": 100,
        "supply_shock": False,
        "populations":   [600, 1200, 400, 900, 700],
        "severities":    [0.2, 0.7,  0.9, 0.5, 0.6],
        "initial_defs":  [12,  20,   8,   15,  12 ],
        "road_blocked":  [False, False, True, False, True],
        "lambda_rates":  [0.1, 0.2, 0.3, 0.1, 0.2],
    },
    "hard": {
        "n_zones": 7,
        "total_steps": 8,
        "initial_supply": 180,
        "supply_shock": True,      # at t=8, R ← ⌊R · ρ⌋, ρ ~ Uniform(0.4, 0.6)
        "populations":   [800, 1500, 300, 1100, 600, 950, 400],
        "severities":    [0.3, 0.8,  1.0, 0.6,  0.7, 0.9, 0.5],
        "initial_defs":  [15,  25,   6,   18,   12,  15,  10 ],
        "road_blocked":  [False, True, True, False, True, False, False],
        "lambda_rates":  [0.1, 0.3, 0.4, 0.2, 0.2, 0.3, 0.1],
    },
}

# Reward weights  w1   w2    w3    w4    w5
REWARD_WEIGHTS = (0.4, 0.3, 0.15, 0.20, 0.10)


# ---------------------------------------------------------------------------
# Gini helper
# ---------------------------------------------------------------------------

def gini_coefficient(values: List[float]) -> float:
    """Normalized Gini of coverage-deficit fractions per zone."""
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    s = sum(values)
    abs_diff = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))
    return abs_diff / (2 * n * s)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class HumanitarianAidEnv:
    """
    OpenEnv-compliant humanitarian aid allocation environment.

    Usage
    -----
    env = HumanitarianAidEnv(task="easy", seed=42)
    obs = env.reset()
    result = env.step(Action(zone_id=0, quantity=10, priority="high"))
    current = env.state()
    score = env.grade()
    """

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        if task not in TASK_CONFIGS:
            raise ValueError(f"task must be one of {list(TASK_CONFIGS)}")
        self.task = task
        self.cfg = TASK_CONFIGS[task]
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # episode tracking
        self._zones: List[ZoneState] = []
        self._global: GlobalState = GlobalState(remaining_supply=0, current_step=0, total_steps=0)
        self._total_waste: int = 0
        self._total_sent: int = 0
        self._done: bool = False
        self._prev_gini: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset episode and return initial observation."""
        cfg = self.cfg
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

        self._zones = [
            ZoneState(
                zone_id=i,
                population=cfg["populations"][i],
                severity=cfg["severities"][i],
                deficit=cfg["initial_defs"][i],
                covered=(cfg["initial_defs"][i] == 0),
                road_blocked=cfg["road_blocked"][i],
                lambda_rate=cfg["lambda_rates"][i],
            )
            for i in range(cfg["n_zones"])
        ]
        self._global = GlobalState(
            remaining_supply=cfg["initial_supply"],
            current_step=0,
            total_steps=cfg["total_steps"],
        )
        self._total_waste = 0
        self._total_sent = 0
        self._done = False

        # initial gini for bonus tracking
        coverage_fracs = self._coverage_fracs()
        self._prev_gini = gini_coefficient(coverage_fracs)

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Apply action, advance environment, return StepResult."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        g = self._global
        zones = self._zones
        w1, w2, w3, w4, w5 = REWARD_WEIGHTS

        # --- validate / clip action ---
        zone_id = int(action.zone_id)
        if zone_id < 0 or zone_id >= len(zones):
            zone_id = 0
        q = max(0, min(int(action.quantity), g.remaining_supply))

        # road-blocked zones: only high priority (airdrop) allowed
        z = zones[zone_id]
        if z.road_blocked and action.priority != "high":
            q = 0  # delivery fails silently, supply not consumed

        # --- pre-step metrics ---
        prev_deficits = [z.deficit for z in zones]
        prev_coverage = [z.covered for z in zones]

        # --- apply allocation ---
        actual_q = min(q, z.deficit)      # can't use more than deficit
        waste = max(0, q - z.deficit)     # oversend waste

        if q > 0:
            g.remaining_supply -= q
            self._total_sent += q
            self._total_waste += waste
            z.deficit = max(0, z.deficit - q)

        # --- stochastic new-need arrival δ_i ~ Poisson(λ_i · σ_i) for each zone ---
        for zone in zones:
            new_need = int(self._np_rng.poisson(zone.lambda_rate * zone.severity))
            zone.deficit = zone.deficit + new_need
            zone.covered = (zone.deficit == 0)

        # --- supply shock (hard task, t=8) ---
        if self.cfg["supply_shock"] and g.current_step == 7 and not g.supply_shock_applied:
            rho = self._rng.uniform(0.4, 0.6)
            g.remaining_supply = math.floor(g.remaining_supply * rho)
            g.supply_shock_applied = True
            g.active_event_flags += 1

        g.current_step += 1

        # --- compute dense reward ---
        # w1: coverage × severity bonus (newly covered zones)
        delta_cov_sev = sum(
            zones[i].severity
            for i in range(len(zones))
            if zones[i].covered and not prev_coverage[i]
        )

        # w2: population-weighted deficit reduction
        total_pop = sum(z.population for z in zones)
        delta_pop_help = sum(
            zones[i].population * max(0, prev_deficits[i] - zones[i].deficit) / total_pop
            for i in range(len(zones))
        )

        # w3: resource waste fraction
        waste_frac = (waste / q) if q > 0 else 0.0

        # w4: critical-zone unmet need penalty
        critical_unmet = sum(
            z.deficit for z in zones if z.severity >= 0.8
        ) / max(total_pop, 1)

        # w5: equity (Gini) bonus
        curr_gini = gini_coefficient(self._coverage_fracs())
        delta_gini = max(0.0, self._prev_gini - curr_gini)
        self._prev_gini = curr_gini

        reward_val = (
            w1 * delta_cov_sev
            + w2 * delta_pop_help
            - w3 * waste_frac
            - w4 * critical_unmet
            + w5 * delta_gini
        )

        # --- terminal check ---
        all_covered = all(z.covered for z in zones)
        self._done = (
            g.current_step >= g.total_steps
            or g.remaining_supply <= 0
            or all_covered
        )

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=Reward(
                value=round(reward_val, 4),
                breakdown={
                    "cov_sev": round(w1 * delta_cov_sev, 4),
                    "pop_help": round(w2 * delta_pop_help, 4),
                    "waste_penalty": round(-w3 * waste_frac, 4),
                    "critical_unmet_penalty": round(-w4 * critical_unmet, 4),
                    "gini_bonus": round(w5 * delta_gini, 4),
                },
            ),
            done=self._done,
            info={
                "step": g.current_step,
                "remaining_supply": g.remaining_supply,
                "total_waste": self._total_waste,
                "supply_shock_applied": g.supply_shock_applied,
                "zone_applied": zone_id,
                "quantity_sent": q,
                "waste": waste,
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return full internal state (for debugging/checkpointing)."""
        return {
            "task": self.task,
            "zones": [z.model_dump() for z in self._zones],
            "global": self._global.model_dump(),
            "total_waste": self._total_waste,
            "total_sent": self._total_sent,
            "done": self._done,
            "prev_gini": self._prev_gini,
        }

    def grade(self) -> GraderResult:
        """
        Terminal grader — call after episode ends.
        G = 0.6·survival_rate + 0.3·efficiency + 0.1·time_bonus
        """
        n = len(self._zones)
        covered_count = sum(1 for z in self._zones if z.covered)
        survival_rate = covered_count / n

        efficiency = (
            1.0 - (self._total_waste / self._total_sent)
            if self._total_sent > 0 else 1.0
        )
        efficiency = max(0.0, min(1.0, efficiency))

        # time_bonus: 1.0 if finished before deadline, 0.5 otherwise
        all_covered = all(z.covered for z in self._zones)
        time_bonus = 1.0 if (all_covered and self._global.current_step < self._global.total_steps) else 0.5

        score = 0.6 * survival_rate + 0.3 * efficiency + 0.1 * time_bonus
        score = round(min(1.0, max(0.0, score)), 4)

        return GraderResult(
            score=score,
            survival_rate=round(survival_rate, 4),
            efficiency=round(efficiency, 4),
            time_bonus=time_bonus,
            details={
                "covered_zones": covered_count,
                "total_zones": n,
                "total_sent": self._total_sent,
                "total_waste": self._total_waste,
                "steps_taken": self._global.current_step,
                "supply_shock": self._global.supply_shock_applied,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coverage_fracs(self) -> List[float]:
        """Deficit fraction per zone (1 = fully unmet, 0 = covered)."""
        result = []
        for z in self._zones:
            init_def = self.cfg["initial_defs"][z.zone_id]
            if init_def == 0:
                result.append(0.0)
            else:
                result.append(min(1.0, z.deficit / max(init_def, 1)))
        return result

    def _make_observation(self) -> Observation:
        hint_parts = []
        for z in self._zones:
            if z.road_blocked:
                hint_parts.append(f"Zone {z.zone_id}: BLOCKED (use priority=high for airdrop)")
            elif z.deficit > 0:
                hint_parts.append(f"Zone {z.zone_id}: needs ~{z.deficit} units (sev={z.severity:.1f})")
        feasible_hint = " | ".join(hint_parts) if hint_parts else "All zones covered!"

        return Observation(
            zones=deepcopy(self._zones),
            global_state=deepcopy(self._global),
            feasible_actions_hint=feasible_hint,
        )
