import random
from typing import List

class ProductionSubsystem:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.conveyor_speeds = {"C1": 0.5, "C2": 0.5, "C3": 0.5}
        self.robot_utilisation = {"R1": 0.5, "R2": 0.5}
        self.interlock_triggered = False
        self.interlock_cooldown = 0
        self.faults: List[dict] = []
        self.current_batch = None
        self.paused_batch_id = None
        self.pause_steps = 0

    def tick(self, available_power: float):
        if self.interlock_cooldown > 0:
            self.interlock_cooldown -= 1
            if self.interlock_cooldown == 0:
                self.interlock_triggered = False
            else:
                self.conveyor_speeds = {k: 0.0 for k in self.conveyor_speeds}
                return

        if self.pause_steps > 0:
            self.pause_steps -= 1

        # Clear old faults after a few ticks (auto-recovery)
        self.faults = [f for f in self.faults if self.rng.random() < 0.7]

        # Simulate faults based on speed — higher speed = more fault risk
        for cid, speed in self.conveyor_speeds.items():
            if speed > 0.8 and self.rng.random() < 0.05:
                fault_type = self.rng.choice(["jam", "overload", "sensor_fail"])
                severity = self.rng.uniform(0.3, 0.9)
                self.faults.append({
                    "conveyor_id": cid,
                    "fault_type": fault_type,
                    "severity": severity
                })
                self.conveyor_speeds[cid] = 0.0  # Shutdown on fault
            elif speed > 0.6 and self.rng.random() < 0.02:
                self.faults.append({
                    "conveyor_id": cid,
                    "fault_type": "overload",
                    "severity": 0.3
                })

        # Robot utilisation tracks conveyor speeds
        avg_speed = sum(self.conveyor_speeds.values()) / len(self.conveyor_speeds)
        self.robot_utilisation["R1"] = min(1.0, avg_speed * 1.2 + self.rng.uniform(-0.1, 0.1))
        self.robot_utilisation["R2"] = min(1.0, avg_speed * 0.9 + self.rng.uniform(-0.1, 0.1))
        self.robot_utilisation = {k: max(0.0, min(1.0, v)) for k, v in self.robot_utilisation.items()}

    def current_load(self) -> float:
        """Power draw in kW — each conveyor takes up to 5 kW, robots take 2 kW."""
        return (sum(s * 5.0 for s in self.conveyor_speeds.values()) +
                sum(r * 2.0 for r in self.robot_utilisation.values()))

    def throughput(self) -> float:
        """Orders that can be processed this step."""
        if self.pause_steps > 0:
            return 0.0
        return sum(self.conveyor_speeds.values()) * 2.0

    def avg_speed(self) -> float:
        """Average conveyor speed for quality pressure calculation."""
        return sum(self.conveyor_speeds.values()) / len(self.conveyor_speeds)

    def batch_output(self) -> str:
        return f"batch_{self.rng.randint(1000, 9999)}"

    def set_speed(self, conveyor_id: str, speed: float):
        if conveyor_id in self.conveyor_speeds:
            self.conveyor_speeds[conveyor_id] = max(0.0, min(1.0, speed))

    def trigger_interlock(self):
        self.interlock_triggered = True
        self.interlock_cooldown = 5

    def pause_segment(self, batch_id: str):
        self.paused_batch_id = batch_id
        self.pause_steps = 2
