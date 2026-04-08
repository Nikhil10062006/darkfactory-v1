import random
import math

FIXED_BASE_LOAD = 10.0  # kW constant facility overhead


class EnergySubsystem:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.time_of_day_steps = 0
        self.battery_capacity_kwh = 50.0
        self.battery_charge_kwh = 25.0  # Start at 50% SOC
        self.grid_price = 0.10           # $ per kWh
        self.solar_output_kw = 0.0       # Set by time-of-day curve
        self.cost_so_far = 0.0           # Agent's actual energy cost
        self.step_cost = 0.0
        self.naive_cost = 0.0            # Baseline: no battery mgmt, no solar offset

    def tick(self, production_load_kw: float, warehouse_load_kw: float):
        self.time_of_day_steps += 1

        # Simulated clock: each step = 2 minutes, start at 6 AM
        # 480 steps = 16 hours (6:00 to 22:00), peak at 14:00-17:00 in middle
        hour = 6.0 + (self.time_of_day_steps * 2.0 / 60.0)

        # --- Dynamic grid pricing ---
        if 14 <= hour < 17:
            self.grid_price = 0.30  # Peak pricing (3x)
        elif 12 <= hour < 14 or 17 <= hour < 19:
            self.grid_price = 0.18  # Shoulder pricing
        else:
            self.grid_price = 0.10  # Off-peak

        # --- Solar output: realistic bell curve peaking at noon ---
        if 6 <= hour <= 19:
            solar_fraction = math.sin(math.pi * (hour - 6.0) / 13.0)
            self.solar_output_kw = 8.0 * max(0.0, solar_fraction)
            noise = self.rng.uniform(-0.5, 0.5)
            self.solar_output_kw = max(0.0, self.solar_output_kw + noise)
        else:
            self.solar_output_kw = 0.0

        # --- Total facility load ---
        total_load_kw = FIXED_BASE_LOAD + production_load_kw + warehouse_load_kw

        # --- Naive baseline: pays full grid price for EVERYTHING (no solar, no battery) ---
        # Step duration = 2 min = 1/30 hour
        naive_step_kwh = total_load_kw / 30.0
        self.naive_cost += naive_step_kwh * self.grid_price

        # --- Agent's actual cost: solar offsets grid draw ---
        net_after_solar = max(0.0, total_load_kw - self.solar_output_kw)

        # Agent pays grid for whatever solar doesn't cover
        # Battery is NOT auto-used — agent must explicitly charge/sell
        step_kwh = net_after_solar / 30.0
        self.step_cost = step_kwh * self.grid_price
        self.cost_so_far += self.step_cost

    def available_power(self) -> float:
        return 1000.0

    def charge(self, kwh: float):
        """
        Charge battery from grid at current price.
        Strategic: charge when price is LOW ($0.10), sell when HIGH ($0.30).
        The charge cost is added, but later selling at peak price creates profit.
        """
        amount = min(max(kwh, 0.0), self.battery_capacity_kwh - self.battery_charge_kwh)
        self.battery_charge_kwh += amount
        self.cost_so_far += amount * self.grid_price

    def sell(self, kwh: float):
        """
        Sell battery energy back to grid at current price.
        Revenue is subtracted from cost, creating profit when price is high.
        """
        amount = min(max(kwh, 0.0), self.battery_charge_kwh)
        self.battery_charge_kwh -= amount
        self.cost_so_far -= amount * self.grid_price

    def step_score(self) -> float:
        """
        Score = savings vs naive baseline.
        Naive pays full load * grid_price every step (no solar, no battery).
        Agent benefits from solar offset automatically.
        Agent can further improve score via battery arbitrage (charge cheap, sell expensive).
        """
        if self.naive_cost <= 0:
            return 0.5
        savings_pct = (self.naive_cost - self.cost_so_far) / self.naive_cost
        return min(max(savings_pct, 0.0), 1.0)
