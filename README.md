---
title: DarkFactory-v1
emoji: 🏭
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# DarkFactory-v1: Autonomous Dark Factory Digital Twin

<strong>OpenEnv Environment</strong> — Meta × Hugging Face Hackathon Submission

***

## Environment Description

DarkFactory-v1 is a real-world simulation of an **autonomous dark factory and quick-commerce dark store** — an industrial facility where zero human workers operate on the floor. Every subsystem is causally coupled: one agent action triggers cascading consequences across energy, production, logistics, and quality simultaneously.

### Why This Matters

Dark factories (Amazon Robotics, Zepto, FreshDirect, Siemens) are a multi-billion dollar industry where AI-driven optimization of energy costs, order fulfillment SLAs, cold-chain compliance, and quality control is a genuine operational challenge. No existing RL benchmark models this compound multi-objective optimization problem.

**Key differentiator:** Adjusting conveyor speed to rush a late order dynamically draws more energy (spiking grid costs during peak hours), increases fault probability, raises cold-zone temperatures, and increases defect rates — creating a rich decision landscape that challenges frontier LLMs.

***

## System Architecture

The simulation comprises four causally coupled subsystems:

```
Agent ──step(action)──► Factory Orchestrator ──► Global reward + observation
                               │
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                      ▼                    ▼
  [Energy Mgmt]      [Production Line]        [Warehouse]           [Quality]
  Solar + battery    Conveyors + robots       Order batching        Defect sampling
  Grid price arb.    Speed/power tradeoff     Cold-chain zones      Audit/rework
  Peak demand mgmt   Fault + interlock        SLA deadlines         Speed pressure
```

### Causal Links

| Agent Action | Energy Impact | Production Impact | Warehouse Impact | Quality Impact |
|---|---|---|---|---|
| `set_conveyor_speed(HIGH)` | Power draw ↑ 23% | Throughput ↑, fault risk ↑ | Orders process faster | Defect rate ↑ |
| `charge_battery` | Grid cost ↑ now, saves later | — | — | — |
| `sell_to_grid` | Revenue during peaks | — | — | — |
| `assign_pick_wave` | — | — | Orders move to processing | — |
| `trigger_quality_check` | — | Throughput paused 2 steps | Slower processing | Catches defects |

***

## Action Space (Discrete Composite)

The environment accepts a JSON dictionary with `action_type` and relevant parameters.

| Action Type | Parameters | Description |
|---|---|---|
| `set_conveyor_speed` | `conveyor_id`: "C1"/"C2"/"C3", `speed`: 0.0–1.0 | Adjust conveyor throughput. High speeds risk jams. |
| `charge_battery` | `kwh`: float | Buy grid power to charge battery. Best during off-peak ($0.10/kWh). |
| `sell_to_grid` | `kwh`: float | Sell battery energy back during peak pricing ($0.30/kWh). |
| `assign_pick_wave` | `order_ids`: list[str], `cold_zone`: "ambient"/"chilled"/"frozen" | Batch orders for processing. Match zone to order requirements. |
| `trigger_quality_check` | `batch_id`: str | Pause production to audit. Catches defects but reduces throughput. |
| `noop` | *(none)* | Do nothing. States evolve naturally. |

***

## Observation Space (Structured Dictionary)

Each `step()` returns a JSON observation:

| Field | Type | Range | Description |
|---|---|---|---|
| `step` | int | 0–max_steps | Current step number |
| `grid_price_kwh` | float | 0.10–0.30 | Current electricity price (peaks 14:00–17:00) |
| `solar_output_kw` | float | 0.0–8.5 | Solar generation (bell curve, peaks at noon) |
| `battery_soc_pct` | float | 0–100 | Battery state of charge |
| `conveyor_speeds` | dict | {str: 0.0–1.0} | Speed of C1, C2, C3 conveyors |
| `robot_utilisation` | dict | {str: 0.0–1.0} | Robot arm utilization (R1, R2) |
| `pending_orders` | list[Order] | 0–20 items | Orders awaiting assignment (capped for context) |
| `cold_zone_temps` | dict | {str: float} | Zone temperatures (ambient ~22°C, chilled <8°C, frozen <-15°C) |
| `defect_rate_pct` | float | 0–30 | Current defect rate (increases with speed) |
| `active_faults` | list[Fault] | 0+ items | Hardware faults (jam, overload, sensor_fail) |
| `energy_cost_so_far` | float | 0+ | Cumulative energy spend |
| `orders_completed` | int | 0+ | Total orders delivered |
| `orders_breached_sla` | int | 0+ | Orders that missed deadline |

***

## Reward Design

Composite reward at every step provides partial progress signal (not sparse):

```
R_total = w_energy × energy_score + w_fulfill × fulfillment_score + w_cold × cold_chain_score
        + safety_penalty
```

| Component | Formula | Signal |
|---|---|---|
| `energy_score` | `(naive_cost - agent_cost) / naive_cost` | Higher when agent manages battery/solar vs grid |
| `fulfillment_score` | `0.6 × completion_rate + 0.4 × on_time_rate` | Higher when orders completed before deadline |
| `cold_chain_score` | `1.0 - (cold_breaches / cold_orders)` | Higher when cold zones stay in temp range |
| `quality_score` | `1.0 - (missed_defects / total_batches × 5)` | Higher when quality checks catch defects |
| `safety_penalty` | `-1.0` if interlock triggered | Devastating penalty for safety violations |

***

## Tasks

### Task 1: Energy Management (Easy) — 480 steps
**Scenario:** 8-hour shift. Solar panels generate power on a bell curve. Grid pricing varies (off-peak $0.10, shoulder $0.18, peak $0.30). Battery has 50 kWh capacity.

**Agent goal:** Minimize net energy cost by charging battery during cheap hours and selling during peak.

**Scoring:** `energy_score` only. Noop agent gets ~0.3 from solar offset alone.

***

### Task 2: Order Fulfillment (Medium) — 120 steps
**Scenario:** 2-hour rush. 100 initial orders, 10 more every 10 steps. 30% require cold-chain. Varied deadlines (8–30 steps).

**Agent goal:** Batch orders into pick waves, assign correct cold zones, meet SLA deadlines.

**Scoring:** `0.6 × fulfillment + 0.4 × cold_chain`. Must balance speed vs accuracy.

***

### Task 3: Crisis Response (Hard) — 480 steps
**Scenario:** Full 8-hour shift. At step 40, a compound crisis fires:
- Solar drops to 0 (cloud cover)
- Grid price spikes to $0.30
- Conveyor C3 develops a fault (capacity drops 60%)
- Frozen zone starts warming (+10°C shock)

**Agent goal:** Simultaneously manage energy costs, reroute production, maintain cold chain, keep SLA breaches below 15%.

**Scoring:** `0.35 × energy + 0.40 × fulfillment + 0.25 × cold_chain`. Safety interlock = instant 0.0.

***

## Baseline Scores

| Task | Noop Agent | Heuristic Agent | Expected Strong LLM |
|---|---|---|---|
| T1 — Energy Management | ~0.30 | ~0.50 | ~0.80 |
| T2 — Order Fulfillment | ~0.40 | ~0.55 | ~0.65 |
| T3 — Crisis Response | ~0.20 | ~0.35 | ~0.45 |

***

## Setup & Usage

### Prerequisites
- Python 3.11+
- Docker (for containerized execution)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.main:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export HF_TOKEN=your_api_key_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

### Docker

```bash
# Build
docker build -t darkfactory-v1 .

# Run server
docker run -p 7860:7860 darkfactory-v1

# Run inference externally
python inference.py
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment. Query param `?task=task1_energy` |
| `/step` | POST | Execute action. Body: JSON action object |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |

***

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Hugging Face / API key | *(required for LLM)* |
| `TASK` | Default task for server | `task1_energy` |
| `ENV_BASE_URL` | Server URL for inference | `http://localhost:7860` |

***

## Project Structure

```
darkfactory-v1/
├── server/
│   ├── main.py              # FastAPI endpoints (/reset, /step, /state)
│   ├── env.py               # DarkFactoryEnv (core simulation logic)
│   ├── models.py            # Pydantic models (Observation, Action, Reward)
│   ├── subsystems/
│   │   ├── energy.py        # Solar, battery, grid pricing
│   │   ├── production.py    # Conveyors, robots, faults
│   │   ├── warehouse.py     # Orders, pick waves, cold zones
│   │   └── quality.py       # Defect sampling, quality audits
│   └── tasks/
│       ├── task1_energy.py   # Energy grader
│       ├── task2_orders.py   # Order fulfillment grader
│       └── task3_crisis.py   # Crisis response grader
├── inference.py              # Baseline inference script
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
