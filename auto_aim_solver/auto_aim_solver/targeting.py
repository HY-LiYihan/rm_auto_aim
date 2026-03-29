import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TargetState:
    center: np.ndarray
    velocity: np.ndarray
    yaw: float
    v_yaw: float
    armors_num: int
    radius_1: float
    radius_2: float
    dz: float


def target_state_from_msg(target_msg) -> TargetState:
    return TargetState(
        center=np.array([
            float(target_msg.position.x),
            float(target_msg.position.y),
            float(target_msg.position.z),
        ], dtype=float),
        velocity=np.array([
            float(target_msg.velocity.x),
            float(target_msg.velocity.y),
            float(target_msg.velocity.z),
        ], dtype=float),
        yaw=float(target_msg.yaw),
        v_yaw=float(target_msg.v_yaw),
        armors_num=max(1, int(target_msg.armors_num)),
        radius_1=float(target_msg.radius_1),
        radius_2=float(target_msg.radius_2),
        dz=float(target_msg.dz),
    )


def predict_target_state(state: TargetState, delta_t: float) -> TargetState:
    return TargetState(
        center=state.center + state.velocity * delta_t,
        velocity=state.velocity.copy(),
        yaw=state.yaw + state.v_yaw * delta_t,
        v_yaw=state.v_yaw,
        armors_num=state.armors_num,
        radius_1=state.radius_1,
        radius_2=state.radius_2,
        dz=state.dz,
    )


def reconstruct_armor_points(state: TargetState) -> List[np.ndarray]:
    points: List[np.ndarray] = []
    is_current_pair = True

    for i in range(state.armors_num):
        yaw_i = state.yaw + i * (2.0 * math.pi / state.armors_num)
        if state.armors_num == 4:
            radius = state.radius_1 if is_current_pair else state.radius_2
            z = state.center[2] if is_current_pair else state.center[2] + state.dz
            is_current_pair = not is_current_pair
        else:
            radius = state.radius_1
            z = state.center[2]

        x = state.center[0] - radius * math.cos(yaw_i)
        y = state.center[1] - radius * math.sin(yaw_i)
        points.append(np.array([x, y, z], dtype=float))

    return points


def select_target_point(points: List[np.ndarray], launch_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if not points:
        return None
    if launch_pos is None:
        return points[0]
    return min(points, key=lambda p: float(np.linalg.norm(p - launch_pos)))
