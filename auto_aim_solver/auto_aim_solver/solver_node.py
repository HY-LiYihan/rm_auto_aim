import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from sensor_msgs.msg import CameraInfo
from auto_aim_interfaces.msg import AutoAimCmd, Target
from venom_serial_driver.msg import GameStatus
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .ballistics import BallisticCalculator, BallisticParams
from .delay import DelayConfig, compute_data_age_s, compute_delta_t_s
from .targeting import (
    predict_target_state,
    reconstruct_armor_points,
    select_target_point,
    target_state_from_msg,
)


class BallisticSolver(Node):
    """Standalone ballistic solver node for auto-aim integration."""

    def __init__(self):
        super().__init__('ballistic_solver')

        self.declare_parameter('mass', 0.0032)
        self.declare_parameter('radius', 0.0085)
        self.declare_parameter('drag_coeff', 0.47)
        self.declare_parameter('air_density', 1.225)
        self.declare_parameter('initial_speed', 28.0)
        self.declare_parameter('launch_frame', 'launcher_link')
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('update_frequency', 30.0)
        self.declare_parameter('target_topic', '/tracker/target')
        self.declare_parameter('target_timeout', 0.2)
        self.declare_parameter('speed_topic', '/game_status')
        self.declare_parameter('use_live_speed', True)
        self.declare_parameter('speed_timeout', 0.5)
        self.declare_parameter('min_live_speed', 5.0)
        self.declare_parameter('auto_fire', False)
        self.declare_parameter('solver.min_pitch', -0.35)
        self.declare_parameter('solver.max_pitch', 0.8)
        self.declare_parameter('solver.pitch_samples', 36)
        self.declare_parameter('solver.max_iterations', 18)
        self.declare_parameter('solver.max_time', 5.0)
        self.declare_parameter('solver.ground_z', 0.0)
        self.declare_parameter('timing.pipeline_delay', 0.02)
        self.declare_parameter('timing.control_delay', 0.015)
        self.declare_parameter('auto_aim_topic', '/auto_aim')
        self.declare_parameter('solution_topic', '/auto_aim/gimbal_cmd')
        self.declare_parameter('aim_point_topic', '/auto_aim/aim_point')
        self.declare_parameter('detect_topic', '/auto_aim/detect')
        self.declare_parameter('track_topic', '/auto_aim/track')
        self.declare_parameter('fire_topic', '/auto_aim/fire')
        self.declare_parameter('distance_topic', '/auto_aim/distance')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('camera_frame', 'camera_optical_frame')

        self.update_params()
        self.target_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.target_sub = self.create_subscription(
            Target, self.target_topic, self.target_callback, self.target_qos)
        self.speed_sub = self.create_subscription(
            GameStatus, self.speed_topic, self.game_status_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.solution_pub = self.create_publisher(Vector3Stamped, self.solution_topic, 10)
        self.aim_point_pub = self.create_publisher(PointStamped, self.aim_point_topic, 10)
        self.auto_aim_pub = self.create_publisher(AutoAimCmd, self.auto_aim_topic, 10)
        self.detect_pub = self.create_publisher(Bool, self.detect_topic, 10)
        self.track_pub = self.create_publisher(Bool, self.track_topic, 10)
        self.fire_pub = self.create_publisher(Bool, self.fire_topic, 10)
        self.distance_pub = self.create_publisher(Float32, self.distance_topic, 10)

        self.latest_target = None
        self.latest_target_time = None
        self.latest_target_state = None
        self.latest_live_speed = None
        self.latest_live_speed_time = None
        self.camera_frame = self.get_parameter('camera_frame').value
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.timer = self.create_timer(1.0 / self.update_frequency, self.solve_once)
        self.get_logger().info('BallisticSolver initialized.')

    def update_params(self):
        self.mass = self.get_parameter('mass').value
        self.radius = self.get_parameter('radius').value
        self.drag_coeff = self.get_parameter('drag_coeff').value
        self.rho = self.get_parameter('air_density').value
        self.v0 = self.get_parameter('initial_speed').value
        self.launch_frame = self.get_parameter('launch_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.update_frequency = self.get_parameter('update_frequency').value
        self.target_topic = self.get_parameter('target_topic').value
        self.target_timeout = self.get_parameter('target_timeout').value
        self.speed_topic = self.get_parameter('speed_topic').value
        self.use_live_speed = self.get_parameter('use_live_speed').value
        self.speed_timeout = self.get_parameter('speed_timeout').value
        self.min_live_speed = self.get_parameter('min_live_speed').value
        self.auto_fire = self.get_parameter('auto_fire').value
        self.min_pitch = self.get_parameter('solver.min_pitch').value
        self.max_pitch = self.get_parameter('solver.max_pitch').value
        self.pitch_samples = self.get_parameter('solver.pitch_samples').value
        self.max_iterations = self.get_parameter('solver.max_iterations').value
        self.solver_max_time = self.get_parameter('solver.max_time').value
        self.ground_z = self.get_parameter('solver.ground_z').value
        self.pipeline_delay_s = self.get_parameter('timing.pipeline_delay').value
        self.control_delay_s = self.get_parameter('timing.control_delay').value
        self.auto_aim_topic = self.get_parameter('auto_aim_topic').value
        self.solution_topic = self.get_parameter('solution_topic').value
        self.aim_point_topic = self.get_parameter('aim_point_topic').value
        self.detect_topic = self.get_parameter('detect_topic').value
        self.track_topic = self.get_parameter('track_topic').value
        self.fire_topic = self.get_parameter('fire_topic').value
        self.distance_topic = self.get_parameter('distance_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        configured_camera_frame = self.get_parameter('camera_frame').value
        if configured_camera_frame:
            self.camera_frame = configured_camera_frame

        self.dt = 1.0 / max(self.update_frequency, 1.0)

        params = BallisticParams(
            mass=self.mass,
            radius=self.radius,
            drag_coeff=self.drag_coeff,
            air_density=self.rho,
            min_pitch=self.min_pitch,
            max_pitch=self.max_pitch,
            pitch_samples=int(self.pitch_samples),
            max_iterations=int(self.max_iterations),
            solver_max_time=self.solver_max_time,
            ground_z=self.ground_z,
            dt=self.dt,
        )
        self.ballistic_calculator = BallisticCalculator(params, self.get_current_muzzle_speed)
        self.delay_config = DelayConfig(
            pipeline_delay_s=self.pipeline_delay_s,
            control_delay_s=self.control_delay_s,
        )

    def target_callback(self, msg):
        self.latest_target = msg
        self.latest_target_time = self.get_clock().now()
        self.latest_target_state = target_state_from_msg(msg)

    def game_status_callback(self, msg):
        if msg.initial_speed >= self.min_live_speed:
            self.latest_live_speed = float(msg.initial_speed)
            self.latest_live_speed_time = self.get_clock().now()

    def camera_info_callback(self, msg):
        if msg.k[0] > 0 and msg.k[4] > 0:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
        elif msg.p[0] > 0 and msg.p[5] > 0:
            self.fx = float(msg.p[0])
            self.fy = float(msg.p[5])
            self.cx = float(msg.p[2])
            self.cy = float(msg.p[6])

        if not self.camera_frame and msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def target_is_fresh(self):
        if self.latest_target is None or self.latest_target_time is None:
            return False
        age = (self.get_clock().now() - self.latest_target_time).nanoseconds / 1e9
        return age <= self.target_timeout

    def get_current_muzzle_speed(self):
        if not self.use_live_speed:
            return self.v0
        if self.latest_live_speed is None or self.latest_live_speed_time is None:
            return self.v0
        age = (self.get_clock().now() - self.latest_live_speed_time).nanoseconds / 1e9
        if age > self.speed_timeout:
            return self.v0
        return self.latest_live_speed

    def get_launch_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.launch_frame, rclpy.time.Time())
            pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ], dtype=float)
            x, y, z, w = (
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            )
            rot = np.array([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ], dtype=float)
            return pos, rot
        except TransformException:
            return None, None

    def get_frame_transform(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            trans = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ], dtype=float)
            x, y, z, w = (
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            )
            rot = np.array([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ], dtype=float)
            return rot, trans
        except TransformException:
            return None, None

    def project_point_to_image(self, point_world):
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None
        if not self.camera_frame:
            return None

        rot, trans = self.get_frame_transform(self.camera_frame, self.map_frame)
        if rot is None:
            return None

        point_camera = rot @ point_world + trans
        z = float(point_camera[2])
        if z <= 1e-6:
            return None

        u = self.fx * float(point_camera[0]) / z + self.cx
        v = self.fy * float(point_camera[1]) / z + self.cy
        return u, v, z

    def solve_once(self):
        has_target = self.target_is_fresh()
        is_tracking = has_target and self.latest_target is not None and bool(self.latest_target.tracking)
        auto_aim_msg = AutoAimCmd()
        auto_aim_msg.header.stamp = self.get_clock().now().to_msg()
        auto_aim_msg.header.frame_id = self.map_frame
        auto_aim_msg.detected = bool(has_target)
        auto_aim_msg.tracking = bool(is_tracking)
        auto_aim_msg.fire = False
        auto_aim_msg.distance = 0.0
        auto_aim_msg.proj_x = 0
        auto_aim_msg.proj_y = 0

        self.detect_pub.publish(Bool(data=has_target))
        self.track_pub.publish(Bool(data=is_tracking))
        self.fire_pub.publish(Bool(data=False))

        if not has_target or not is_tracking or self.latest_target_state is None:
            self.auto_aim_pub.publish(auto_aim_msg)
            return

        launch_pos, launch_rot = self.get_launch_pose()
        if launch_pos is None:
            self.auto_aim_pub.publish(auto_aim_msg)
            return

        current_candidates = reconstruct_armor_points(self.latest_target_state)
        coarse_target = select_target_point(current_candidates, launch_pos)
        if coarse_target is None:
            self.auto_aim_pub.publish(auto_aim_msg)
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        sample_sec = self.latest_target_time.nanoseconds / 1e9
        data_age_s = compute_data_age_s(now_sec, sample_sec)
        flight_guess_s = self.ballistic_calculator.estimate_flight_time_s(coarse_target, launch_pos)
        delta_t = compute_delta_t_s(data_age_s, flight_guess_s, self.delay_config)

        predicted_state = predict_target_state(self.latest_target_state, delta_t)
        predicted_candidates = reconstruct_armor_points(predicted_state)
        target_point = select_target_point(predicted_candidates, launch_pos)
        if target_point is None:
            self.auto_aim_pub.publish(auto_aim_msg)
            return

        solution = self.ballistic_calculator.solve_ballistic_arc(target_point, launch_pos, launch_rot)
        if solution is None:
            self.auto_aim_pub.publish(auto_aim_msg)
            return

        target_distance = float(np.linalg.norm(target_point - launch_pos))

        aim_msg = PointStamped()
        aim_msg.header.stamp = self.get_clock().now().to_msg()
        aim_msg.header.frame_id = self.map_frame
        aim_msg.point.x = float(target_point[0])
        aim_msg.point.y = float(target_point[1])
        aim_msg.point.z = float(target_point[2])
        self.aim_point_pub.publish(aim_msg)

        reprojection = self.project_point_to_image(target_point)
        if reprojection is not None:
            auto_aim_msg.proj_x = int(round(reprojection[0]))
            auto_aim_msg.proj_y = int(round(reprojection[1]))

        sol_msg = Vector3Stamped()
        sol_msg.header = aim_msg.header
        sol_msg.vector.x = float(solution['yaw'])
        sol_msg.vector.y = float(solution['pitch'])
        sol_msg.vector.z = float(solution['flight_time'])
        self.solution_pub.publish(sol_msg)

        auto_aim_msg.pitch = float(solution['pitch'])
        auto_aim_msg.yaw = float(solution['yaw'])
        auto_aim_msg.distance = target_distance
        auto_aim_msg.fire = bool(self.auto_fire and self.latest_target.tracking)

        self.distance_pub.publish(Float32(data=target_distance))
        self.fire_pub.publish(Bool(data=auto_aim_msg.fire))
        self.auto_aim_pub.publish(auto_aim_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BallisticSolver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
