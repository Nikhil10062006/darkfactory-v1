"""
Task 1: Energy Management Grader (Easy)
Evaluates how much the agent saved on energy costs vs naive baseline.
"""

def grade_t1(trajectory: list) -> float:
    """
    Grade energy arbitrage performance.
    
    Args:
        trajectory: list of observation dicts from each step
    
    Returns:
        Score 0.0-1.0 based on energy cost savings vs naive baseline
    """
    if not trajectory:
        return 0.0

    final_step = trajectory[-1]
    agent_cost = final_step.get("energy_cost_so_far", 0.0)

    # Naive baseline: pay full grid price for all power, no solar/battery
    FIXED_LOAD = 10.0  # kW base load
    naive_cost = 0.0
    for step in trajectory:
        grid_price = step.get("grid_price_kwh", 0.10)
        # Naive pays for full load every minute (1/60 hour)
        naive_cost += FIXED_LOAD / 60.0 * grid_price

    if naive_cost <= 0:
        return 0.0

    savings_pct = (naive_cost - agent_cost) / naive_cost
    return min(max(savings_pct, 0.0), 1.0)
