from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any
import os, json

from .models import (
    DarkFactoryAction, DarkFactoryObservation, DarkFactoryReward,
    NoOp, SetConveyorSpeed, ChargeBattery, SellToGrid, AssignPickWave, TriggerQualityCheck
)
from .env import DarkFactoryEnv

app = FastAPI(title="DarkFactory-v1", description="OpenEnv dark factory simulation")
TASK = os.getenv("TASK", "task1_energy")

# Global environment instance
env = DarkFactoryEnv(task=TASK)


def parse_action(data: dict) -> DarkFactoryAction:
    """
    Parse raw JSON dict into a typed DarkFactoryAction.
    Falls back to NoOp for any malformed input instead of raising 422.
    """
    action_type = data.get("action_type", "noop")

    try:
        if action_type == "set_conveyor_speed":
            return SetConveyorSpeed(
                action_type="set_conveyor_speed",
                conveyor_id=data.get("conveyor_id", "C1"),
                speed=float(data.get("speed", 0.5))
            )
        elif action_type == "charge_battery":
            return ChargeBattery(
                action_type="charge_battery",
                kwh=float(data.get("kwh", 5.0))
            )
        elif action_type == "sell_to_grid":
            return SellToGrid(
                action_type="sell_to_grid",
                kwh=float(data.get("kwh", 5.0))
            )
        elif action_type == "assign_pick_wave":
            order_ids = data.get("order_ids", [])
            if isinstance(order_ids, str):
                order_ids = [order_ids]
            cold_zone = data.get("cold_zone", "ambient")
            if cold_zone not in ("ambient", "chilled", "frozen"):
                cold_zone = "ambient"
            return AssignPickWave(
                action_type="assign_pick_wave",
                order_ids=order_ids,
                cold_zone=cold_zone
            )
        elif action_type == "trigger_quality_check":
            return TriggerQualityCheck(
                action_type="trigger_quality_check",
                batch_id=str(data.get("batch_id", "batch_0000"))
            )
        else:
            return NoOp(action_type="noop")
    except Exception:
        return NoOp(action_type="noop")


@app.post("/reset")
def reset(task: str = None):
    """Reset the environment. Optionally specify a task."""
    global env, TASK
    if task:
        TASK = task
    env = DarkFactoryEnv(task=TASK)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(request: Request):
    """
    Execute one step. Accepts raw JSON — gracefully handles malformed actions
    by falling back to noop instead of returning 422 errors.
    """
    try:
        body = await request.json()
    except Exception:
        body = {"action_type": "noop"}

    action = parse_action(body)

    if env.done:
        obs = env._get_observation()
        reward = DarkFactoryReward(
            total=0.0, energy_component=0.0, fulfillment_component=0.0,
            cold_chain_component=0.0, quality_component=0.0,
            safety_penalty=0.0, info={"done": True}
        )
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": True,
            "info": {"message": "Episode already finished. Call /reset to start a new episode."}
        }

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    """Return current environment state."""
    return env.state()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "task": TASK, "step": env.step_count, "done": env.done}
