"""
DarkFactory-v1 Inference Script
Follows mandatory STDOUT format: [START] [STEP] [END]
"""
import os, json, sys, time
import openai
from openai import OpenAI
import requests

# --- Mandatory environment variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "darkfactory-v1"

# Per-task step limits matching openenv.yaml
TASK_CONFIG = {
    "task1_energy": {"max_steps": 120, "name": "Energy Management"},
    "task2_orders": {"max_steps": 120, "name": "Order Fulfillment"},
    "task3_crisis": {"max_steps": 120, "name": "Crisis Response"},
}

TASKS = list(TASK_CONFIG.keys())

# --- LLM Client Setup ---
fake_agent = not API_KEY

client = None
if API_KEY:
    try:
        import httpx
        http_client = httpx.Client(proxy=None)
    except ImportError:
        http_client = None
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        http_client=http_client
    )

# --- System Prompts ---
SYSTEM_PROMPT = """You are an autonomous dark factory controller AI agent managing a zero-human industrial facility.

You receive a JSON observation containing:
- grid_price_kwh: current electricity price (0.10 off-peak, 0.18 shoulder, 0.30 peak)
- solar_output_kw: current solar panel generation
- battery_soc_pct: battery state of charge (0-100%)
- conveyor_speeds: {"C1": 0-1.0, "C2": 0-1.0, "C3": 0-1.0}
- pending_orders: list of orders with deadlines counting down
- cold_zone_temps: {"ambient": ~22, "chilled": target<8, "frozen": target<-15}
- active_faults: any hardware faults requiring attention
- energy_cost_so_far: cumulative energy spend

You must output EXACTLY ONE action as a JSON object. Available actions:

1. {"action_type": "charge_battery", "kwh": 5.0}
   - Buy grid power to charge battery. Best when grid_price_kwh is LOW (0.10).

2. {"action_type": "sell_to_grid", "kwh": 5.0}
   - Sell battery energy back. Best when grid_price_kwh is HIGH (0.30).

3. {"action_type": "set_conveyor_speed", "conveyor_id": "C1", "speed": 0.8}
   - Adjust conveyor speed (0.0-1.0). Higher speed = more throughput but more faults and energy.

4. {"action_type": "assign_pick_wave", "order_ids": ["O0001","O0002"], "cold_zone": "ambient"}
   - Assign pending orders to processing. cold_zone: "ambient", "chilled", or "frozen".
   - Match cold_zone to order requirements (requires_cold orders need "chilled" or "frozen").

5. {"action_type": "trigger_quality_check", "batch_id": "batch_1234"}
   - Pause production to catch defects. Reduces throughput but improves quality score.

6. {"action_type": "noop"}
   - Do nothing this step.

STRATEGY TIPS:
- Charge battery when price is 0.10, sell when price is 0.30
- Keep conveyor speeds at 0.5-0.7 to balance throughput vs fault risk
- Assign orders before their deadline_steps reaches 0
- Cold orders (requires_cold=true) need "chilled" or "frozen" zone
- Watch active_faults — reduce speed on faulted conveyors

Output ONLY the JSON action. No explanation, no markdown, no code blocks."""


def get_action_from_llm(obs_json: str, task_name: str, step_n: int, last_reward: float) -> str:
    """Get action from LLM. Returns JSON string."""
    if fake_agent:
        # Smart fallback: basic heuristic instead of always noop
        try:
            obs = json.loads(obs_json)
            price = obs.get("grid_price_kwh", 0.10)
            soc = obs.get("battery_soc_pct", 50)
            pending = obs.get("pending_orders", [])

            if price <= 0.10 and soc < 80:
                return json.dumps({"action_type": "charge_battery", "kwh": 5.0})
            elif price >= 0.30 and soc > 20:
                return json.dumps({"action_type": "sell_to_grid", "kwh": 5.0})
            elif pending:
                # Try to assign first few pending orders
                order_ids = [o["order_id"] for o in pending[:3]]
                cold_needed = any(o.get("requires_cold", False) for o in pending[:3])
                zone = "chilled" if cold_needed else "ambient"
                return json.dumps({"action_type": "assign_pick_wave", "order_ids": order_ids, "cold_zone": zone})
            else:
                return json.dumps({"action_type": "noop"})
        except Exception:
            return '{"action_type": "noop"}'

    # Real LLM call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_name} | Step: {step_n} | Last reward: {last_reward:.2f}\n\nObservation:\n{obs_json}"}
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=200,
        temperature=0.0
    )
    return (resp.choices[0].message.content or "").strip()


def run_task(task_name: str):
    config = TASK_CONFIG[task_name]
    max_steps = config["max_steps"]

    # Reset environment for this task
    try:
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task_name}, timeout=30)
        reset_resp.raise_for_status()
        obs = reset_resp.json()
    except Exception as e:
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
        print(f"!! Failed to reset environment: {e}", flush=True)
        return

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    step_n = 0
    last_reward = 0.0

    while step_n < max_steps:
        step_n += 1

        # Get action
        try:
            action_str = get_action_from_llm(json.dumps(obs), task_name, step_n, last_reward)

            if not fake_agent:
                time.sleep(1.6)  # Rate limit: stay under 40 RPM
        except openai.RateLimitError:
            print(f"!! Rate limit hit at step {step_n}. Sleeping 10s...", flush=True)
            time.sleep(10)
            continue  # Retry same step
        except Exception as e:
            print(f"!! API Error at step {step_n}: {e}. Retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        # Parse action JSON
        try:
            clean_str = action_str.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean_str)
            # Normalize: some models use "action" instead of "action_type"
            if "action" in action and "action_type" not in action:
                action["action_type"] = action.pop("action")
        except Exception:
            action = {"action_type": "noop"}

        # Step environment
        try:
            res = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=10).json()

            obs = res.get("observation", res)
            reward_data = res.get("reward", {})
            reward = reward_data.get("total", 0.0) if isinstance(reward_data, dict) else float(reward_data)
            done = res.get("done", False)
            error = res.get("info", {}).get("error", None)

            rewards.append(reward)
            last_reward = reward
            done_str = "true" if done else "false"
            error_str = str(error) if error else "null"

            print(f"[STEP] step={step_n} action={json.dumps(action)} "
                  f"reward={reward:.2f} done={done_str} "
                  f"error={error_str}", flush=True)

            if done:
                break

        except Exception as e:
            print(f"!! Env Error at step {step_n}: {e}. Retrying step...", flush=True)
            time.sleep(2)
            continue

    # --- End of Task ---
    if not rewards:
        rewards = [0.0]

    # Final score: average of all step rewards (clamped to [0, 1])
    score = sum(rewards) / len(rewards)
    score = min(max(score, 0.0), 1.0)

    success = score >= 0.3
    success_str = "true" if success else "false"
    reward_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={success_str} steps={len(rewards)} score={score:.2f} rewards={reward_str}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
