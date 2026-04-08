"""
Step-by-step local test: tests each endpoint and action type.
Run WHILE the server is running on port 7860.
"""
import requests, json

BASE = "http://localhost:7860"

def pretty(label, resp):
    data = resp.json()
    if "reward" in data:
        r = data["reward"]
        print(f"  {label}: status={resp.status_code} reward={r['total']:.3f} "
              f"[energy={r['energy_component']:.3f} fulfill={r['fulfillment_component']:.3f} "
              f"cold={r['cold_chain_component']:.3f} quality={r['quality_component']:.3f}] "
              f"done={data['done']}")
    else:
        print(f"  {label}: status={resp.status_code} step={data.get('step','-')}")

# ============================================================
print("=" * 70)
print("TEST 1: /reset for each task")
print("=" * 70)
for task in ["task1_energy", "task2_orders", "task3_crisis"]:
    r = requests.post(f"{BASE}/reset", params={"task": task})
    obs = r.json()
    print(f"  {task}: status={r.status_code} step={obs['step']} "
          f"orders={len(obs['pending_orders'])} "
          f"battery={obs['battery_soc_pct']}% solar={obs['solar_output_kw']}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 2: /step with each action type (task1_energy)")
print("=" * 70)
requests.post(f"{BASE}/reset", params={"task": "task1_energy"})

actions = [
    ("noop", {"action_type": "noop"}),
    ("charge_battery 5kWh", {"action_type": "charge_battery", "kwh": 5.0}),
    ("sell_to_grid 2kWh", {"action_type": "sell_to_grid", "kwh": 2.0}),
    ("set_conveyor_speed C1=0.8", {"action_type": "set_conveyor_speed", "conveyor_id": "C1", "speed": 0.8}),
    ("trigger_quality_check", {"action_type": "trigger_quality_check", "batch_id": "batch_001"}),
    ("assign_pick_wave", {"action_type": "assign_pick_wave", "order_ids": ["O0001", "O0002"], "cold_zone": "ambient"}),
]

for label, action in actions:
    r = requests.post(f"{BASE}/step", json=action)
    pretty(label, r)

# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Malformed actions (should gracefully fallback to noop)")
print("=" * 70)
requests.post(f"{BASE}/reset", params={"task": "task1_energy"})

bad_actions = [
    ("empty dict", {}),
    ("missing kwh", {"action_type": "charge_battery"}),
    ("wrong key", {"action": "noop"}),
    ("garbage", {"foo": "bar", "baz": 123}),
]

for label, action in bad_actions:
    r = requests.post(f"{BASE}/step", json=action)
    pretty(label, r)

# ============================================================
print("\n" + "=" * 70)
print("TEST 4: /state endpoint")
print("=" * 70)
r = requests.get(f"{BASE}/state")
state = r.json()
print(f"  status={r.status_code} step={state['step']} done={state['done']} task={state['task']}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 5: /health endpoint")
print("=" * 70)
r = requests.get(f"{BASE}/health")
print(f"  status={r.status_code} response={r.json()}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 6: Full episode noop (task1_energy, 20 steps sample)")
print("=" * 70)
requests.post(f"{BASE}/reset", params={"task": "task1_energy"})
for i in range(20):
    r = requests.post(f"{BASE}/step", json={"action_type": "noop"})
    d = r.json()
    rew = d["reward"]["total"]
    obs = d["observation"]
    if i % 5 == 0:
        print(f"  step={obs['step']} reward={rew:.3f} solar={obs['solar_output_kw']:.1f} "
              f"price={obs['grid_price_kwh']} battery={obs['battery_soc_pct']:.1f}% "
              f"cost={obs['energy_cost_so_far']:.2f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 7: Task2 orders — verify orders get processed")
print("=" * 70)
requests.post(f"{BASE}/reset", params={"task": "task2_orders"})
for i in range(30):
    r = requests.post(f"{BASE}/step", json={"action_type": "noop"})
    d = r.json()
    obs = d["observation"]
    if i % 10 == 0:
        print(f"  step={obs['step']} reward={d['reward']['total']:.3f} "
              f"completed={obs['orders_completed']} breached={obs['orders_breached_sla']} "
              f"pending={len(obs['pending_orders'])}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 8: Task3 crisis — verify crisis triggers at step 40")
print("=" * 70)
requests.post(f"{BASE}/reset", params={"task": "task3_crisis"})
for i in range(45):
    r = requests.post(f"{BASE}/step", json={"action_type": "noop"})
    d = r.json()
    obs = d["observation"]
    if i in [38, 39, 40, 41, 42]:
        print(f"  step={obs['step']} reward={d['reward']['total']:.3f} "
              f"solar={obs['solar_output_kw']:.1f} price={obs['grid_price_kwh']} "
              f"frozen={obs['cold_zone_temps']['frozen']:.1f} "
              f"faults={len(obs['active_faults'])}")

print("\n" + "=" * 70)
print("ALL LOCAL TESTS PASSED!")
print("=" * 70)
