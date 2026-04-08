"""
Task 3: Crisis Response Grader (Hard)
Evaluates multi-objective performance under compound stress event.
"""

def grade_t3(trajectory: list) -> float:
    """
    Grade crisis response performance.
    
    Args:
        trajectory: list of observation dicts from each step
    
    Returns:
        Score 0.0-1.0 based on energy, fulfillment, and cold chain under crisis
    """
    if not trajectory:
        return 0.0

    final_step = trajectory[-1]

    # Safety interlock check — instant disqualification
    active_faults = final_step.get("active_faults", [])
    if any(f.get("fault_type") == "safety_interlock" for f in active_faults):
        return 0.0

    # Energy component
    agent_cost = final_step.get("energy_cost_so_far", 0.0)
    FIXED_LOAD = 10.0
    naive_cost = sum(
        step.get("grid_price_kwh", 0.10) * FIXED_LOAD / 60.0
        for step in trajectory
    )
    energy_score = 0.0
    if naive_cost > 0:
        savings_pct = (naive_cost - agent_cost) / naive_cost
        energy_score = min(max(savings_pct, 0.0), 1.0)

    # SLA component
    orders_completed = final_step.get("orders_completed", 0)
    orders_breached = final_step.get("orders_breached_sla", 0)
    total = orders_completed + orders_breached + len(final_step.get("pending_orders", []))
    sla_score = 1.0 - min(orders_breached / max(total, 1), 1.0) if total > 0 else 0.5

    # Cold chain component
    cold_temps = final_step.get("cold_zone_temps", {})
    temp_penalty = 0.0
    if cold_temps.get("frozen", -18.0) > -15.0:
        temp_penalty += 0.5
    if cold_temps.get("chilled", 4.0) > 8.0:
        temp_penalty += 0.3
    cold_score = max(0.0, 1.0 - temp_penalty)

    return 0.35 * energy_score + 0.40 * sla_score + 0.25 * cold_score
