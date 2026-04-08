import random

class QualitySubsystem:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.base_defect_rate = 0.02       # 2% baseline defect rate
        self.defect_rate_pct = 0.02        # Current effective defect rate
        self.checked_batches = set()
        self.total_batches = 0
        self.defective_batches = 0
        self.caught_defects = 0            # Defects caught by quality checks
        self.missed_defects = 0            # Defects shipped to customer
        self.speed_pressure = 0.0          # Accumulated from high conveyor speeds

    def tick(self, batch_id: str):
        if batch_id:
            self.total_batches += 1

            # Defect rate increases with speed pressure
            self.defect_rate_pct = min(
                self.base_defect_rate + self.speed_pressure * 0.05,
                0.30  # Cap at 30%
            )

            # Roll for defect
            if self.rng.random() < self.defect_rate_pct:
                self.defective_batches += 1
                if batch_id in self.checked_batches:
                    # Quality check caught this — rework instead of ship
                    self.caught_defects += 1
                else:
                    # Defect shipped to customer
                    self.missed_defects += 1

        # Speed pressure decays slowly each tick
        self.speed_pressure = max(0.0, self.speed_pressure * 0.95)

    def add_speed_pressure(self, avg_speed: float):
        """Called by env to reflect conveyor speed impact on quality."""
        if avg_speed > 0.6:
            self.speed_pressure += (avg_speed - 0.6) * 0.5

    def check(self, batch_id: str):
        """Mark a batch for quality inspection."""
        self.checked_batches.add(batch_id)

    def step_score(self) -> float:
        """
        Quality score: based on how many defects were caught vs missed.
        1.0 = no missed defects, 0.0 = many missed defects.
        """
        if self.total_batches == 0:
            return 1.0
        if self.defective_batches == 0:
            return 1.0
        catch_rate = self.caught_defects / max(self.defective_batches, 1)
        miss_penalty = self.missed_defects / max(self.total_batches, 1)
        # Reward catching defects, penalize misses
        score = 1.0 - (miss_penalty * 5.0)  # 5x amplifier for missed defects
        return max(0.0, min(1.0, score))
