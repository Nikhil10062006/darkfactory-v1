import random
from typing import Dict, Any, Tuple

from .models import (
    DarkFactoryObservation, DarkFactoryAction, DarkFactoryReward, Order, Fault
)
from .subsystems.energy import EnergySubsystem
from .subsystems.production import ProductionSubsystem
from .subsystems.warehouse import WarehouseSubsystem
from .subsystems.quality import QualitySubsystem

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

        # Load task-specific configurations
        if self.task == "task1_energy":
            self.max_steps = 480
            # Task 1: Some background orders to create energy load variation
            self.warehouse.spawn_orders(20)
        elif self.task == "task2_orders":
            self.warehouse.spawn_orders(100)
            self.max_steps = 120
        elif self.task == "task3_crisis":
            self.warehouse.spawn_orders(50)
            self.max_steps = 480
        else:
            self.max_steps = 480

        self.done = False
        self.crisis_triggered = False

    def reset(self) -> DarkFactoryObservation:
        self.rng = random.Random(self.seed)
        self._init_subsystems()
        return self._get_observation()

    def step(self, action: DarkFactoryAction):
        # 1. Apply agent action
        self._apply_action(action)

        # 2. Task 3 Crisis trigger at step 40
        if self.task == "task3_crisis" and self.step_count == 40 and not self.crisis_triggered:
            self.crisis_triggered = True
            self.energy.solar_output_kw = 0.0
            self.energy.grid_price = 0.30
            self.production.conveyor_speeds["C3"] = 0.2
            self.warehouse.cold_zone_temps["frozen"] += 10.0  # Warming emergency
            self.production.faults.append({
                "conveyor_id": "C3",
                "fault_type": "overload",
                "severity": 0.8
            })

        # 3. Periodic order spawns for tasks 2 and 3
        if self.task in ["task2_orders", "task3_crisis"] and self.step_count > 0 and self.step_count % 10 == 0:
            self.warehouse.spawn_orders(10)

        # 4. Feed conveyor speed pressure into quality system
        self.quality.add_speed_pressure(self.production.avg_speed())

        # 5. Tick all subsystems (causal coupling)
        self.energy.tick(self.production.current_load(), self.warehouse.refrigeration_load())
        self.production.tick(self.energy.available_power())
        self.warehouse.tick(self.production.throughput())
        self.quality.tick(self.production.batch_output())

        # 6. Advance step counter and check termination
        self.step_count += 1
        self.done = self._check_done()

        # 7. Compute reward
        reward = self._compute_reward()
        return self._get_observation(), reward, self.done, {"step": self.step_count}

    def state(self) -> dict:
        obs = self._get_observation()
        state_dict = obs.model_dump()
        state_dict["done"] = self.done
        state_dict["task"] = self.task
        return state_dict

    def _get_observation(self) -> DarkFactoryObservation:
        return DarkFactoryObservation(
            step=self.step_count,
            grid_price_kwh=self.energy.grid_price,
            solar_output_kw=self.energy.solar_output_kw,
            battery_soc_pct=(self.energy.battery_charge_kwh / self.energy.battery_capacity_kwh) * 100.0,
            conveyor_speeds=self.production.conveyor_speeds.copy(),
            robot_utilisation=self.production.robot_utilisation.copy(),
            pending_orders=[
                Order(
                    order_id=o.order_id,
                    sku_ids=o.sku_ids,
                    priority=o.priority,
                    requires_cold=o.requires_cold,
                    deadline_steps=o.deadline_steps
                ) for o in self.warehouse.pending_orders[:20]  # Cap at 20 to keep observation size manageable
            ],
            cold_zone_temps=self.warehouse.cold_zone_temps.copy(),
            defect_rate_pct=self.quality.defect_rate_pct * 100.0,
            active_faults=[
                Fault(conveyor_id=f["conveyor_id"], fault_type=f["fault_type"], severity=f["severity"])
                for f in self.production.faults
            ],
            energy_cost_so_far=round(self.energy.cost_so_far, 4),
            orders_completed=self.warehouse.orders_completed,
            orders_breached_sla=self.warehouse.orders_breached_sla
        )

    def _apply_action(self, action: DarkFactoryAction):
        if not action or action.action_type == "noop":
            return

        action_type = action.action_type
        if action_type == "set_conveyor_speed":
            self.production.set_speed(action.conveyor_id, action.speed)
        elif action_type == "charge_battery":
            self.energy.charge(action.kwh)
        elif action_type == "sell_to_grid":
            self.energy.sell(action.kwh)
        elif action_type == "assign_pick_wave":
            self.warehouse.assign_wave(action.order_ids, action.cold_zone)
        elif action_type == "trigger_quality_check":
            self.quality.check(action.batch_id)
            self.production.pause_segment(action.batch_id)

    def _check_done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if self.task == "task3_crisis" and self.production.interlock_triggered:
            return True
        return False

    def _compute_reward(self) -> DarkFactoryReward:
        energy_score = self.energy.step_score()
        fulfillment_score = self.warehouse.step_score()
        cold_chain_score = self.warehouse.cold_chain_score()
        quality_score = self.quality.step_score()
        safety_penalty = -1.0 if self.production.interlock_triggered else 0.0

        if self.task == "task1_energy":
            total = energy_score
        elif self.task == "task2_orders":
            total = 0.6 * fulfillment_score + 0.4 * cold_chain_score
        elif self.task == "task3_crisis":
            if self.production.interlock_triggered:
                total = 0.0
            else:
                total = 0.35 * energy_score + 0.40 * fulfillment_score + 0.25 * cold_chain_score
        else:
            total = (
                0.30 * energy_score +
                0.35 * fulfillment_score +
                0.20 * cold_chain_score +
                0.15 * quality_score +
                safety_penalty
            )

        return DarkFactoryReward(
            total=max(0.0, min(1.0, total)),
            energy_component=round(energy_score, 4),
            fulfillment_component=round(fulfillment_score, 4),
            cold_chain_component=round(cold_chain_score, 4),
            quality_component=round(quality_score, 4),
            safety_penalty=safety_penalty,
            info={
                "step": self.step_count,
                "battery_soc": round(self.energy.battery_charge_kwh, 2),
                "grid_price": self.energy.grid_price,
                "pending_count": len(self.warehouse.pending_orders),
                "active_faults": len(self.production.faults)
            }
        )
