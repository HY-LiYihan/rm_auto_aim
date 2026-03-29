import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class BallisticParams:
    mass: float
    radius: float
    drag_coeff: float
    air_density: float
    min_pitch: float
    max_pitch: float
    pitch_samples: int
    max_iterations: int
    solver_max_time: float
    ground_z: float
    dt: float


class BallisticCalculator:
    def __init__(self, params: BallisticParams, muzzle_speed_provider: Callable[[], float]):
        self.params = params
        self.get_muzzle_speed = muzzle_speed_provider
        self.area = math.pi * (self.params.radius ** 2)

    def estimate_flight_time_s(self, target_world: np.ndarray, launch_pos: np.ndarray) -> float:
        speed = max(self.get_muzzle_speed(), 1e-3)
        distance = float(np.linalg.norm(target_world - launch_pos))
        return distance / speed

    @staticmethod
    def world_to_launch(point_world: np.ndarray, launch_pos: np.ndarray, launch_rot: np.ndarray) -> np.ndarray:
        return launch_rot.T @ (point_world - launch_pos)

    @staticmethod
    def direction_from_angles(yaw: float, pitch: float, launch_rot: np.ndarray) -> np.ndarray:
        local_dir = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ], dtype=float)
        world_dir = launch_rot @ local_dir
        return world_dir / np.linalg.norm(world_dir)

    def calculate_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        v_mag = np.linalg.norm(velocity)
        if v_mag > 0.001:
            drag_force = 0.5 * self.params.air_density * (v_mag ** 2) * self.params.drag_coeff * self.area
            drag_vec = -(velocity / v_mag) * drag_force
        else:
            drag_vec = np.zeros(3)
        gravity_vec = np.array([0.0, 0.0, -self.params.mass * 9.81])
        return (drag_vec + gravity_vec) / self.params.mass

    def evaluate_pitch(
        self,
        pitch: float,
        yaw: float,
        launch_pos: np.ndarray,
        launch_rot: np.ndarray,
        target_world: np.ndarray,
        target_range: float,
    ) -> Optional[Dict[str, float]]:
        muzzle_speed = self.get_muzzle_speed()
        vel = self.direction_from_angles(yaw, pitch, launch_rot) * muzzle_speed
        pos = launch_pos.copy()
        prev_pos = pos.copy()
        prev_range = 0.0
        flight_time = 0.0
        aim_dir_xy = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
        target_local = self.world_to_launch(target_world, launch_pos, launch_rot)

        max_steps = max(1, int(self.params.solver_max_time / self.params.dt))
        for _ in range(max_steps):
            prev_pos = pos.copy()
            acc = self.calculate_acceleration(vel)
            vel += acc * self.params.dt
            pos += vel * self.params.dt
            flight_time += self.params.dt

            local_pos = self.world_to_launch(pos, launch_pos, launch_rot)
            horizontal_range = float(np.dot(local_pos[:2], aim_dir_xy))

            if pos[2] <= self.params.ground_z and horizontal_range < target_range:
                return None
            if horizontal_range >= target_range:
                denom = horizontal_range - prev_range
                ratio = 0.0 if abs(denom) < 1e-6 else (target_range - prev_range) / denom
                ratio = min(max(ratio, 0.0), 1.0)
                hit_pos = prev_pos + ratio * (pos - prev_pos)
                hit_local = self.world_to_launch(hit_pos, launch_pos, launch_rot)
                z_error = float(hit_local[2] - target_local[2])
                miss_distance = float(np.linalg.norm(hit_pos - target_world))
                return {
                    'pitch': float(pitch),
                    'yaw': float(yaw),
                    'flight_time': float(flight_time - self.params.dt + ratio * self.params.dt),
                    'z_error': z_error,
                    'miss_distance': miss_distance,
                    'hit_pos': hit_pos,
                }
            prev_range = horizontal_range
        return None

    def solve_ballistic_arc(
        self,
        target_world: np.ndarray,
        launch_pos: np.ndarray,
        launch_rot: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        target_local = self.world_to_launch(target_world, launch_pos, launch_rot)
        target_range = float(np.linalg.norm(target_local[:2]))
        if target_range < 1e-4:
            return None

        yaw = math.atan2(target_local[1], target_local[0])
        sample_pitches = np.linspace(self.params.min_pitch, self.params.max_pitch, int(self.params.pitch_samples))
        valid_results = []
        for pitch in sample_pitches:
            result = self.evaluate_pitch(float(pitch), yaw, launch_pos, launch_rot, target_world, target_range)
            if result is not None:
                valid_results.append(result)
        if not valid_results:
            return None

        best = min(valid_results, key=lambda item: abs(item['z_error']))
        sign_change = None
        for left, right in zip(valid_results[:-1], valid_results[1:]):
            if left['z_error'] == 0.0:
                sign_change = (left, left)
                break
            if left['z_error'] * right['z_error'] < 0.0:
                sign_change = (left, right)
                break
        if sign_change is None:
            return best

        low_pitch = sign_change[0]['pitch']
        high_pitch = sign_change[1]['pitch']
        low_error = sign_change[0]['z_error']
        refined = best
        for _ in range(int(self.params.max_iterations)):
            mid_pitch = 0.5 * (low_pitch + high_pitch)
            mid_result = self.evaluate_pitch(mid_pitch, yaw, launch_pos, launch_rot, target_world, target_range)
            if mid_result is None:
                break
            refined = mid_result
            if abs(mid_result['z_error']) < abs(best['z_error']):
                best = mid_result
            if abs(mid_result['z_error']) < 1e-3:
                return mid_result
            if low_error * mid_result['z_error'] <= 0.0:
                high_pitch = mid_pitch
            else:
                low_pitch = mid_pitch
                low_error = mid_result['z_error']
        return best if abs(best['z_error']) <= abs(refined['z_error']) else refined
