# DarkFactory-v1 — OpenEnv Hackathon Project Plan

> **Hackathon:** Meta × Hugging Face OpenEnv Round 1  
> **Deadline:** 8 April 2026, 11:59 PM IST  
> **Team:** Nikhil Agarwal · Swarit Agrawal(Lead) · Aryan Sonone  
> **Submission:** HF Spaces URL pasted before deadline by team lead

---

## 1. Concept Overview

**DarkFactory-v1** is a unified simulation of an autonomous dark factory / quick-commerce dark store — a real industrial system where zero human workers operate on the floor and every subsystem is causally coupled to every other.

The key differentiator: **one agent action triggers cascading consequences across 4 subsystems simultaneously.** This is the "world model" property that no existing OpenEnv environment has, and it is what will carry the 30% Real-World Utility score.

### Why this wins

| Criterion | Weight | Why DarkFactory scores high |
|---|---|---|
| Real-world utility | 30% | Dark factories (Zepto, Amazon Robotics, Siemens) are a $multi-billion industry. No OpenEnv environment models this. |
| Task & grader quality | 25% | 3 tasks with genuine difficulty progression. T3 is unsolvable by naive LLMs. |
| Environment design | 20% | Rich partial-reward signal at every step. Causal coupling creates non-trivial dynamics. |
| Code quality | 15% | FastAPI + Pydantic + Docker. Straightforward to validate. |
| Creativity & novelty | 10% | First quick-commerce dark store in OpenEnv. Cold-chain + energy + robotics = novel combo. |

---

## 2. System Architecture

### Four causally coupled subsystems

```
Agent ──step(action)──► Factory Orchestrator ──► Global reward + state
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                      ▼                    ▼
  [Energy Mgmt]      [Production Line]        [Dark Store]         [Quality]
  Solar + battery    Robot scheduling         Order wave batching   Defect sampling
  Grid price arb.    Speed/power tradeoff     Pick-path optim.     Audit trail
  Peak demand shift  Fault + interlock        Cold-chain zones     Reject / rework
```

### Causal links (what makes this a world model)

- `set_conveyor_speed(HIGH)` → throughput ↑, motor power draw ↑ 23%, fault probability ↑, may breach demand-charge threshold
- `assign_cold_zone(pharma)` → refrigeration load ↑, pulls from battery SOC, reduces grid sell-back opportunity
- `trigger_quality_check(batch)` → halts line temporarily, reduces throughput, but prevents defective shipments
- `charge_battery(kwh)` → reduces available grid capacity for machines, but unlocks peak-hour sell-back

---

## 3. Data Models (Pydantic)

### Observation

```python
from pydantic import BaseModel
from typing import Dict, List, Optional

class Order(BaseModel):
    order_id: str
    sku_ids: List[str]
    priority: str          # "express" | "standard"
    requires_cold: bool
    deadline_steps: int    # steps remaining before SLA breach

class Fault(BaseModel):
    conveyor_id: str
    fault_type: str        # "jam" | "overload" | "sensor_fail"
    severity: float        # 0.0–1.0

class DarkFactoryObservation(BaseModel):
    step: int
    grid_price_kwh: float              # current electricity price (spikes during peak)
    solar_output_kw: float             # 0.0 when cloudy
    battery_soc_pct: float             # 0–100
    conveyor_speeds: Dict[str, float]  # {"C1": 0.8, "C2": 1.0, "C3": 0.5}
    robot_utilisation: Dict[str, float]# {"R1": 0.9, "R2": 0.4}
    pending_orders: List[Order]
    cold_zone_temps: Dict[str, float]  # {"pharma": 4.2, "frozen": -18.1}
    defect_rate_pct: float
    active_faults: List[Fault]
    energy_cost_so_far: float
    orders_completed: int
    orders_breached_sla: int
```

### Action

```python
from typing import Literal, Union

class SetConveyorSpeed(BaseModel):
    action_type: Literal["set_conveyor_speed"]
    conveyor_id: str       # "C1" | "C2" | "C3"
    speed: float           # 0.0–1.0 (fraction of max speed)

class ChargeBattery(BaseModel):
    action_type: Literal["charge_battery"]
    kwh: float             # amount to charge (capped by available grid headroom)

class SellToGrid(BaseModel):
    action_type: Literal["sell_to_grid"]
    kwh: float             # discharge battery and sell

class AssignPickWave(BaseModel):
    action_type: Literal["assign_pick_wave"]
    order_ids: List[str]   # batch of orders to pick in this wave
    cold_zone: str         # "ambient" | "chilled" | "frozen"

class TriggerQualityCheck(BaseModel):
    action_type: Literal["trigger_quality_check"]
    batch_id: str          # halts that conveyor segment for N steps

class NoOp(BaseModel):
    action_type: Literal["noop"]

DarkFactoryAction = Union[
    SetConveyorSpeed, ChargeBattery, SellToGrid,
    AssignPickWave, TriggerQualityCheck, NoOp
]
```

### Reward

```python
class DarkFactoryReward(BaseModel):
    total: float                  # 0.0–1.0 composite (what grader uses)
    energy_component: float       # cost savings vs naive baseline
    fulfillment_component: float  # orders on time / total orders
    cold_chain_component: float   # 1.0 - (breaches / total cold orders)
    quality_component: float      # 1.0 - defect_rate
    safety_penalty: float         # 0 normally; -1.0 if safety interlock violated
    info: dict                    # step-level diagnostics for debugging
```

### Composite reward formula

```
R_total = (0.30 × energy) + (0.35 × fulfillment) + (0.20 × cold_chain) + (0.15 × quality)
        + safety_penalty

where safety_penalty = -1.0 if any conveyor runs without E-stop interlock active
```

---

## 4. The Three Tasks

### Task 1 — Energy arbitrage (Easy)

**Scenario:** 8-hour simulation. Static production load (conveyors fixed at medium speed). Solar panel on roof. Battery with 50 kWh capacity. Grid price varies over 8h with one known peak window (14:00–17:00).

**Agent goal:** Charge battery during cheap grid hours, sell back during peak, minimise net energy cost.

**Grader:**
```python
def grade_t1(trajectory) -> float:
    naive_cost = sum(step.grid_price * FIXED_LOAD for step in trajectory)
    agent_cost = trajectory[-1].energy_cost_so_far
    savings_pct = (naive_cost - agent_cost) / naive_cost
    return min(max(savings_pct, 0.0), 1.0)
```

**Expected baseline score:** ~0.35 (random agent does slightly better than naive by accident)  
**Expected strong agent score:** ~0.80

---

### Task 2 — Order wave fulfillment (Medium)

**Scenario:** 2-hour window, 100 incoming orders. 30% require cold-chain (chilled or frozen). Orders arrive in waves every 10 sim-steps (mimicking Zepto app demand spikes). Three conveyor lines, each with different throughput capacity.

**Agent must:** Batch orders into pick waves, assign correct cold zone, sequence conveyors to meet SLA deadlines, avoid cold-chain temperature breaches.

**Grader:**
```python
def grade_t2(trajectory) -> float:
    on_time_ratio = trajectory[-1].orders_completed / TOTAL_ORDERS
    cold_safe_ratio = 1.0 - (cold_breaches / cold_orders)
    return 0.6 * on_time_ratio + 0.4 * cold_safe_ratio
```

**Expected baseline score:** ~0.25  
**Expected strong agent score:** ~0.65

---

### Task 3 — Compound crisis (Hard)

**Scenario:** Full factory simulation (8h). At step 40, a heatwave event fires:
- Solar output drops to 0 (cloud cover)
- Grid price spikes to 3× normal for 2 hours
- Conveyor C3 develops a fault (reduces capacity 60%)
- Cold zone "frozen" starts warming (compressor stress) — agent must reduce load or breach temp

**Agent must:** Simultaneously manage energy costs, reroute conveyor traffic from C3 to C1/C2, maintain frozen zone temp, keep SLA breach rate below 15%, and NOT trigger the safety interlock (which shuts down all conveyors for 5 steps — a devastating penalty).

**Grader:**
```python
def grade_t3(trajectory) -> float:
    if trajectory.safety_interlock_triggered:
        return 0.0   # disqualified
    energy_score = compute_energy_savings(trajectory)
    sla_score = 1.0 - min(trajectory[-1].orders_breached_sla / TOTAL_ORDERS, 1.0)
    cold_score = 1.0 - (frozen_breaches / total_frozen_orders)
    return 0.35 * energy_score + 0.40 * sla_score + 0.25 * cold_score
```

**Expected baseline score:** ~0.05 (most LLMs fail catastrophically on the compound event)  
**Expected strong agent score:** ~0.45

---

## 5. File Structure

```
darkfactory-v1/
│
├── server/
│   ├── main.py              # FastAPI app — /reset, /step, /state endpoints
│   ├── env.py               # DarkFactoryEnv class (core simulation logic)
│   ├── subsystems/
│   │   ├── energy.py        # Solar, battery, grid price simulation
│   │   ├── production.py    # Conveyors, robot arms, fault model
│   │   ├── warehouse.py     # Order generation, pick waves, cold zones
│   │   └── quality.py       # Defect sampling, audit trail
│   ├── tasks/
│   │   ├── task1_energy.py  # Task config + grader
│   │   ├── task2_orders.py  # Task config + grader
│   │   └── task3_crisis.py  # Task config + grader
│   ├── models.py            # All Pydantic models (Observation, Action, Reward)
│   └── config.py            # Simulation constants
│
├── inference.py             # MANDATORY: root-level inference script
├── openenv.yaml             # Environment metadata
├── Dockerfile               # Containerised server
├── requirements.txt
└── README.md
```

---

## 6. Core Environment Implementation

### `server/env.py` (skeleton)

```python
import random
from models import DarkFactoryObservation, DarkFactoryAction, DarkFactoryReward
from subsystems.energy import EnergySubsystem
from subsystems.production import ProductionSubsystem
from subsystems.warehouse import WarehouseSubsystem
from subsystems.quality import QualitySubsystem

class DarkFactoryEnv:
    def __init__(self, task: str = "task1_energy", seed: int = 42):
        self.task = task
        self.seed = seed
        self.rng = random.Random(seed)
        self._init_subsystems()

    def _init_subsystems(self):
        self.energy = EnergySubsystem(self.rng)
        self.production = ProductionSubsystem(self.rng)
        self.warehouse = WarehouseSubsystem(self.rng)
        self.quality = QualitySubsystem(self.rng)
        self.step_count = 0
        self.done = False

    def reset(self) -> DarkFactoryObservation:
        self.rng = random.Random(self.seed)
        self._init_subsystems()
        return self._get_observation()

    def step(self, action: DarkFactoryAction):
        # 1. Apply action to relevant subsystem
        self._apply_action(action)
        # 2. Tick all subsystems (causal coupling happens here)
        self.energy.tick(self.production.current_load(), self.warehouse.refrigeration_load())
        self.production.tick(self.energy.available_power())
        self.warehouse.tick(self.production.throughput())
        self.quality.tick(self.production.batch_output())
        # 3. Check task termination
        self.step_count += 1
        self.done = self._check_done()
        # 4. Compute reward
        reward = self._compute_reward()
        return self._get_observation(), reward, self.done, {}

    def state(self) -> dict:
        return self._get_observation().model_dump()

    def _apply_action(self, action: DarkFactoryAction):
        match action.action_type:
            case "set_conveyor_speed":
                self.production.set_speed(action.conveyor_id, action.speed)
            case "charge_battery":
                self.energy.charge(action.kwh)
            case "sell_to_grid":
                self.energy.sell(action.kwh)
            case "assign_pick_wave":
                self.warehouse.assign_wave(action.order_ids, action.cold_zone)
            case "trigger_quality_check":
                self.quality.check(action.batch_id)
                self.production.pause_segment(action.batch_id)
            case "noop":
                pass

    def _compute_reward(self) -> DarkFactoryReward:
        # Causal composite reward — partial signals at every step
        energy_score = self.energy.step_score()
        fulfillment_score = self.warehouse.step_score()
        cold_chain_score = self.warehouse.cold_chain_score()
        quality_score = self.quality.step_score()
        safety_penalty = -1.0 if self.production.interlock_triggered else 0.0
        total = (
            0.30 * energy_score +
            0.35 * fulfillment_score +
            0.20 * cold_chain_score +
            0.15 * quality_score +
            safety_penalty
        )
        return DarkFactoryReward(
            total=max(0.0, min(1.0, total)),
            energy_component=energy_score,
            fulfillment_component=fulfillment_score,
            cold_chain_component=cold_chain_score,
            quality_component=quality_score,
            safety_penalty=safety_penalty,
            info={"step": self.step_count}
        )
```

### `server/main.py` (FastAPI)

```python
from fastapi import FastAPI
from models import DarkFactoryAction
from env import DarkFactoryEnv
import os

app = FastAPI()
TASK = os.getenv("TASK", "task1_energy")
env = DarkFactoryEnv(task=TASK)

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: DarkFactoryAction):
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}

@app.get("/state")
def state():
    return env.state()
```

---

## 7. `openenv.yaml`

```yaml
name: darkfactory-v1
version: "1.0.0"
description: >
  Autonomous dark factory / quick-commerce dark store simulation.
  Four causally coupled subsystems: energy management, production line,
  warehouse order fulfillment, and quality control.
  Designed to benchmark agent multi-objective reasoning under compound stress events.
tags:
  - openenv
  - manufacturing
  - logistics
  - energy
  - real-world
tasks:
  - id: task1_energy
    name: "Energy Arbitrage"
    difficulty: easy
    max_steps: 480
  - id: task2_orders
    name: "Order Wave Fulfillment"
    difficulty: medium
    max_steps: 120
  - id: task3_crisis
    name: "Compound Crisis"
    difficulty: hard
    max_steps: 480
action_space: discrete_composite
observation_space: structured_dict
reward_range: [0.0, 1.0]
```

---

## 8. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY openenv.yaml .

ENV TASK=task1_energy
EXPOSE 7860

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### `requirements.txt`

```
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.0
openai==1.25.0
numpy==1.26.4
```

---

## 9. `inference.py` (Root Level — Mandatory)

```python
"""
DarkFactory-v1 Inference Script
Follows mandatory STDOUT format: [START] [STEP] [END]
"""
import os, json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

import requests

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an autonomous factory controller agent.
You receive a JSON observation of a dark factory state and must output ONE action as JSON.
Available actions: set_conveyor_speed, charge_battery, sell_to_grid,
assign_pick_wave, trigger_quality_check, noop.
Reason about energy cost, order deadlines, cold-chain temps, and robot faults.
Output ONLY valid JSON matching the action schema. No explanation."""

TASKS = ["task1_energy", "task2_orders", "task3_crisis"]
MAX_STEPS = 120  # well within 20-min runtime limit

def run_task(task_name: str):
    obs = requests.post(f"{ENV_BASE_URL}/reset",
                        params={"task": task_name}).json()
    print(f"[START] task={task_name} env=darkfactory-v1 model={MODEL_NAME}")

    rewards = []
    for step_n in range(1, MAX_STEPS + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(obs)}
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=200, temperature=0.0
        )
        action_str = resp.choices[0].message.content.strip()

        try:
            action = json.loads(action_str)
        except Exception:
            action = {"action_type": "noop"}

        result = requests.post(f"{ENV_BASE_URL}/step", json=action).json()
        obs      = result["observation"]
        reward   = result["reward"]["total"]
        done     = result["done"]
        error    = result.get("info", {}).get("error", None)

        rewards.append(reward)
        print(f"[STEP] step={step_n} action={action_str[:60]} "
              f"reward={reward:.2f} done={str(done).lower()} "
              f"error={error or 'null'}")

        if done:
            break

    success = rewards[-1] >= 0.5
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_n} rewards={reward_str}")


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
```

---

## 10. Implementation Schedule

### Day 1 — Thursday 3 April: Foundation

- [ ] `openenv init darkfactory-v1` — scaffold project
- [ ] Implement `models.py` — all Pydantic models (Observation, Action, Reward)
- [ ] Implement `server/subsystems/energy.py` — grid price curve, solar, battery SOC
- [ ] Implement `server/env.py` — reset(), state(), basic step() with energy-only
- [ ] Implement `server/main.py` — FastAPI endpoints
- [ ] Verify `curl -X POST localhost:7860/reset` returns 200
- [ ] Write Task 1 grader in `server/tasks/task1_energy.py`

**End-of-day goal:** Task 1 is playable locally. `openenv validate` passes.

---

### Day 2 — Friday 4 April: Production + Warehouse

- [ ] Implement `server/subsystems/production.py` — conveyors, robot arms, fault model, safety interlock
- [ ] Implement `server/subsystems/warehouse.py` — order generation, pick waves, cold zones
- [ ] Wire causal coupling in `env._apply_action()` and `env.step()` tick sequence
- [ ] Write Task 2 grader in `server/tasks/task2_orders.py`
- [ ] Run Task 1 + Task 2 end-to-end with a `noop` agent. Verify scores are in [0.0, 1.0]
- [ ] Build `Dockerfile` — `docker build && docker run` must work

**End-of-day goal:** Tasks 1 and 2 complete. Docker confirmed working.

---

### Day 3 — Saturday 5 April: Crisis + Quality + Baseline

- [ ] Implement `server/subsystems/quality.py` — defect sampling, batch audit
- [ ] Implement Task 3 compound crisis event (heatwave trigger at step 40)
- [ ] Write Task 3 grader in `server/tasks/task3_crisis.py`
- [ ] Write `inference.py` — test all three tasks, confirm [START]/[STEP]/[END] stdout format
- [ ] Verify inference runtime < 20 min on simulated 2 vCPU / 8 GB machine
- [ ] Push to Hugging Face Space — verify `openenv validate` passes remotely

**End-of-day goal:** All 3 tasks working. Baseline scores recorded. HF Space live.

---

### Day 4 — Sunday 6 April: Polish + README + Buffer

- [ ] Write `README.md` (environment description, action/obs space, task descriptions, baseline scores, setup instructions)
- [ ] Write `openenv.yaml` with all required metadata
- [ ] Add Gradio dashboard to HF Space (live factory floor viz — battery gauge, order queue, conveyor status)
- [ ] Run pre-submission validation script — all 3/3 checks must pass
- [ ] Stress-test: run inference 3× and confirm scores are reproducible (variance < 0.02)
- [ ] Buffer time for bug fixes

---

### Day 5–6 — Monday/Tuesday 7–8 April: Final checks + Submit

- [ ] Final `openenv validate` on remote HF Space URL
- [ ] Final `docker build && docker run` on clean machine
- [ ] Confirm inference.py is at root level, not inside `server/`
- [ ] Confirm env vars `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` are documented
- [ ] Team lead (Aryan) pastes HF Spaces URL in submission form before 8 Apr 11:59 PM IST

---

## 11. Pre-Submission Checklist

All items below are **disqualification gates** — the automated validator checks each one.

- [ ] HF Space URL returns HTTP 200 on `POST /reset`
- [ ] `openenv.yaml` present and valid
- [ ] Pydantic models typed correctly (Observation, Action, Reward)
- [ ] `step()`, `reset()`, `state()` all respond correctly
- [ ] `Dockerfile` builds without error (`docker build .`)
- [ ] `inference.py` at root level (not inside `server/`)
- [ ] `inference.py` uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars
- [ ] 3+ tasks enumerable, each grader returns score in [0.0, 1.0]
- [ ] Inference runtime < 20 minutes
- [ ] Baseline scores recorded and reproducible (run 3× — variance < 0.02)
- [ ] `README.md` includes: description, action space, observation space, task descriptions, setup instructions, baseline scores

---

## 12. Environment Variables Reference

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Hugging Face API key | `hf_xxxxxxxxxxxx` |
| `TASK` | Which task to load in the server | `task1_energy` |
| `ENV_BASE_URL` | Base URL of the running server | `http://localhost:7860` |

---

## 13. Expected Baseline Scores

| Task | Naive (noop) | Random agent | Strong LLM | Human upper bound |
|---|---|---|---|---|
| T1 — Energy arbitrage | 0.00 | 0.18 | ~0.80 | ~0.95 |
| T2 — Order wave | 0.00 | 0.15 | ~0.65 | ~0.90 |
| T3 — Compound crisis | 0.00 | 0.05 | ~0.45 | ~0.80 |

*Scores are estimates — record actual values after Day 3 baseline run and paste into README.*

---

## 14. HF Space Gradio Dashboard (Bonus)

Build a live visual on the HF Space landing page (separate from the API endpoints):

- **Battery gauge** — circular arc showing SOC %, colour-coded (green → amber → red)
- **Grid price chart** — last 20 steps, vertical line at current step
- **Order queue** — scrolling list of pending orders with countdown timers
- **Conveyor status** — three bars showing C1/C2/C3 speed and fault indicator
- **Cold zone temps** — three thermometers (ambient / chilled / frozen) with target bands

This is purely for the human review stage (Phase 3 — Meta/HF engineers). It is not required for automated validation. Budget 2–3 hours on Day 4 if time permits.

---

## 15. Key Design Decisions

**Why FastAPI over Flask?** Pydantic v2 integrates natively. Models double as both API schema and internal state. `openenv validate` can ping typed endpoints directly.

**Why seed-based determinism?** Every `reset(seed=42)` produces identical initial state. Graders are therefore deterministic and reproducible across machines — a hard requirement of the judging rubric.

**Why partial reward at every step (not just episode end)?** The judging rubric explicitly penalises sparse reward. The composite formula gives non-zero signal even when no orders complete — energy savings accumulate per-step.

**Why the compound crisis at step 40 (not step 0)?** The agent must demonstrate adaptation — it needs to perform reasonably for 40 steps first, then respond correctly to the crisis. An agent that panics and triggers the safety interlock at step 40 scores 0.0 regardless of earlier performance. This is what makes T3 genuinely hard.

**Quick-commerce specifics (why Zepto/Instamart framing matters):** Quick-commerce (10-min delivery) warehouses operate with extreme time pressure, small grid layouts (~500 SKUs), and cold-chain requirements for pharma and fresh food. This is a real operational challenge that no existing RL benchmark models. Calling it out explicitly in the README and `openenv.yaml` description will resonate with Meta engineers who work on real-world AI deployment.

---

*Last updated: 1 April 2026*
