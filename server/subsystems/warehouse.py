import random
from typing import List, Dict
from ..models import Order

class WarehouseSubsystem:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.pending_orders: List[Order] = []
        self.cold_zone_temps = {"ambient": 22.0, "chilled": 4.0, "frozen": -18.0}
        self.orders_completed = 0
        self.orders_completed_on_time = 0
        self.orders_breached_sla = 0
        self.total_orders = 0
        self.cold_breaches = 0
        self.cold_orders = 0
        self.active_waves: Dict[str, List[Order]] = {"ambient": [], "chilled": [], "frozen": []}

    def tick(self, throughput: float):
        # --- Temperature physics ---
        # Without active cooling, cold zones drift toward ambient (22°C)
        for zone in self.cold_zone_temps:
            if zone == "chilled":
                self.cold_zone_temps[zone] = min(self.cold_zone_temps[zone] + 0.1, 22.0)
            elif zone == "frozen":
                self.cold_zone_temps[zone] = min(self.cold_zone_temps[zone] + 0.2, 22.0)

        # --- Check cold-chain violations ---
        if self.cold_zone_temps["chilled"] > 8.0:  # Exceeds safe chilled range
            self.cold_breaches += 1
        if self.cold_zone_temps["frozen"] > -15.0:  # Exceeds safe frozen range
            self.cold_breaches += 1

        # --- Process active waves based on throughput ---
        processed = int(throughput)

        all_processing = []
        for zone_orders in self.active_waves.values():
            all_processing.extend(zone_orders)

        # Prioritize orders closest to deadline
        all_processing.sort(key=lambda o: o.deadline_steps)

        completed_this_step = all_processing[:processed]
        for order in completed_this_step:
            for zone, lst in self.active_waves.items():
                if order in lst:
                    lst.remove(order)
                    break
            self.orders_completed += 1
            if order.deadline_steps >= 0:
                self.orders_completed_on_time += 1
            else:
                self.orders_breached_sla += 1
            if order.requires_cold:
                # Check if cold chain was maintained
                if order.requires_cold and self.cold_zone_temps.get("frozen", -18.0) > -15.0:
                    pass  # cold breach already tracked above

        # --- Decrement deadlines for ALL pending and active orders ---
        # Pending orders (not yet assigned to waves)
        expired = []
        for order in self.pending_orders:
            order.deadline_steps -= 1
            if order.deadline_steps < -5:  # Grace period of 5 steps past deadline
                expired.append(order)

        # Remove expired orders (without mutating during iteration)
        for order in expired:
            self.pending_orders.remove(order)
            self.orders_breached_sla += 1

        # Active wave orders also tick down
        for zone_orders in self.active_waves.values():
            for order in zone_orders:
                order.deadline_steps -= 1

    def refrigeration_load(self) -> float:
        """Energy cost of running compressors to maintain cold chain."""
        load = 0.0
        if self.cold_zone_temps["chilled"] > 2.0:
            load += 5.0
            self.cold_zone_temps["chilled"] -= 0.2  # Active cooling
        if self.cold_zone_temps["frozen"] > -20.0:
            load += 10.0
            self.cold_zone_temps["frozen"] -= 0.4  # Active cooling
        return load

    def assign_wave(self, order_ids: List[str], cold_zone: str):
        """Move orders from pending queue to a processing wave."""
        if cold_zone not in self.active_waves:
            return
        selected = [o for o in self.pending_orders if o.order_id in order_ids]
        if selected:
            self.active_waves[cold_zone].extend(selected)
            for o in selected:
                if o in self.pending_orders:
                    self.pending_orders.remove(o)

    def spawn_orders(self, count: int):
        for _ in range(count):
            self.total_orders += 1
            is_cold = self.rng.random() < 0.3
            if is_cold:
                self.cold_orders += 1
            deadline = self.rng.choice([8, 12, 20, 30])  # Varied deadlines
            priority = "express" if deadline <= 12 else "standard"
            o = Order(
                order_id=f"O{self.total_orders:04d}",
                sku_ids=[f"SKU{self.rng.randint(1, 500)}" for _ in range(self.rng.randint(1, 5))],
                priority=priority,
                requires_cold=is_cold,
                deadline_steps=deadline
            )
            self.pending_orders.append(o)

    def step_score(self) -> float:
        """Fulfillment score: weighted by on-time completion."""
        if self.total_orders == 0:
            return 1.0
        completion_ratio = self.orders_completed / max(self.total_orders, 1)
        if self.orders_completed == 0:
            return 0.5  # Neutral when no orders exist yet
        on_time_ratio = self.orders_completed_on_time / max(self.orders_completed, 1)
        # Combined: 60% completion rate + 40% on-time rate
        score = 0.6 * completion_ratio + 0.4 * on_time_ratio
        return max(0.0, min(1.0, score))

    def cold_chain_score(self) -> float:
        """Score based on avoiding temperature violations."""
        if self.cold_orders == 0:
            return 1.0
        # Each cold breach is penalized relative to total cold orders
        penalty = min(self.cold_breaches / max(self.cold_orders * 5, 1), 1.0)
        return max(0.0, 1.0 - penalty)
