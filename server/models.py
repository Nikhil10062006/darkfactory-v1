from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Union, Optional, Annotated

class Order(BaseModel):
    order_id: str
    sku_ids: List[str]
    priority: str          # "express" | "standard"
    requires_cold: bool
    deadline_steps: int    # steps remaining before SLA breach

class Fault(BaseModel):
    conveyor_id: str
    fault_type: str        # "jam" | "overload" | "sensor_fail"
    severity: float        # 0.0–1.0

class DarkFactoryObservation(BaseModel):
    step: int
    grid_price_kwh: float              # current electricity price (spikes during peak)
    solar_output_kw: float             # 0.0 when cloudy
    battery_soc_pct: float             # 0–100
    conveyor_speeds: Dict[str, float]  # {"C1": 0.8, "C2": 1.0, "C3": 0.5}
    robot_utilisation: Dict[str, float]# {"R1": 0.9, "R2": 0.4}
    pending_orders: List[Order]
    cold_zone_temps: Dict[str, float]  # {"pharma": 4.2, "frozen": -18.1}
    defect_rate_pct: float
    active_faults: List[Fault]
    energy_cost_so_far: float
    orders_completed: int
    orders_breached_sla: int

class SetConveyorSpeed(BaseModel):
    action_type: Literal["set_conveyor_speed"]
    conveyor_id: str       # "C1" | "C2" | "C3"
    speed: float           # 0.0–1.0 (fraction of max speed)

class ChargeBattery(BaseModel):
    action_type: Literal["charge_battery"]
    kwh: float             # amount to charge (capped by available grid headroom)

class SellToGrid(BaseModel):
    action_type: Literal["sell_to_grid"]
    kwh: float             # discharge battery and sell

class AssignPickWave(BaseModel):
    action_type: Literal["assign_pick_wave"]
    order_ids: List[str]   # batch of orders to pick in this wave
    cold_zone: str         # "ambient" | "chilled" | "frozen"

class TriggerQualityCheck(BaseModel):
    action_type: Literal["trigger_quality_check"]
    batch_id: str          # halts that conveyor segment for N steps

class NoOp(BaseModel):
    action_type: Literal["noop"]

DarkFactoryAction = Annotated[
    Union[
        SetConveyorSpeed, ChargeBattery, SellToGrid,
        AssignPickWave, TriggerQualityCheck, NoOp
    ],
    Field(discriminator="action_type")
]

class DarkFactoryReward(BaseModel):
    total: float                  # 0.0–1.0 composite (what grader uses)
    energy_component: float       # cost savings vs naive baseline
    fulfillment_component: float  # orders on time / total orders
    cold_chain_component: float   # 1.0 - (breaches / total cold orders)
    quality_component: float      # 1.0 - defect_rate
    safety_penalty: float         # 0 normally; -1.0 if safety interlock violated
    info: dict                    # step-level diagnostics for debugging
