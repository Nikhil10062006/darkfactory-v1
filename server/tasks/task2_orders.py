"""
Task 2: Order Fulfillment Grader (Medium)
Evaluates order completion rate and cold-chain compliance.
"""

def grade_t2(trajectory: list) -> float:
    """
    Grade order fulfillment performance.
    
    Args:
        trajectory: list of observation dicts from each step
    
    Returns:
        Score 0.0-1.0 based on fulfillment rate and cold chain safety
    """
    if not trajectory:
        return 0.0

    final_step = trajectory[-1]
    orders_completed = final_step.get("orders_completed", 0)
    orders_breached = final_step.get("orders_breached_sla", 0)

    # Dynamic total: orders spawn in waves
    total_orders = orders_completed + orders_breached + len(final_step.get("pending_orders", []))
    if total_orders == 0:
        return 0.5

    completion_ratio = orders_completed / max(total_orders, 1)
    breach_ratio = orders_breached / max(total_orders, 1)

    # Cold chain evaluation
    cold_temps = final_step.get("cold_zone_temps", {})
    temp_penalty = 0.0
    if cold_temps.get("frozen", -18.0) > -15.0:
        temp_penalty += 0.2
    if cold_temps.get("chilled", 4.0) > 8.0:
        temp_penalty += 0.2

    cold_safe_ratio = max(0.0, 1.0 - temp_penalty)

    fulfillment_score = max(0.0, completion_ratio - breach_ratio * 0.5)
    return min(1.0, 0.6 * fulfillment_score + 0.4 * cold_safe_ratio)
