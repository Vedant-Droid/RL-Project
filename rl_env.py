#!/usr/bin/env python3
"""
Goal-reaching Gym environment for a two-wheel differential drive robot
with a 2D lidar, running in Gazebo (Harmonic) via ROS 2 Jazzy.

Task      : drive from the start pose to a randomly placed goal without colliding.
Observation (n_lidar + 4 values, all scaled to [-1, 1]):
    - n_lidar downsampled lidar ranges (min over each angular bin)
    - distance to goal
    - angle to goal (relative to robot heading)
    - last linear velocity command
    - last angular velocity command
Action    : Box(2) in [-1, 1]
    - a[0] -> linear velocity  in [0, max_linear]   (forward only)
    - a[1] -> angular velocity in [-max_angular, +max_angular]
Reward    : progress toward goal, +100 on reaching it, -100 on collision,
            small penalties for time, spinning and getting close to obstacles.

Goal and robot pose are tracked in the odom frame, relative to the pose the
robot has right after each reset, so it works whether or not odom resets.
"""
import math
import subprocess
import threading
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class _RosIO(Node):
    """Thin ROS 2 node: stores latest scan/odom, publishes /cmd_vel."""

    def __init__(self, use_stamped: bool):
        super().__init__('tb3_rl_env')
        self.use_stamped = use_stamped
        self.lock = threading.Lock()

        self.scan_ranges = None
        self.range_min = 0.0
        self.range_max = 3.5
        self.scan_count = 0

        self.pose = None          # (x, y, yaw) in odom frame
        self.odom_count = 0

        msg_type = TwistStamped if use_stamped else Twist
        self.cmd_pub = self.create_publisher(msg_type, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)

    def _scan_cb(self, msg: LaserScan):
        with self.lock:
            self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
            self.range_min = float(msg.range_min)
            self.range_max = float(msg.range_max)
            self.scan_count += 1

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
            self.odom_count += 1

    def publish_cmd(self, lin: float, ang: float):
        twist = Twist()
        twist.linear.x = float(lin)
        twist.angular.z = float(ang)
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist = twist
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(twist)

    def snapshot(self):
        with self.lock:
            scan = None if self.scan_ranges is None else self.scan_ranges.copy()
            return scan, self.range_min, self.range_max, self.pose

    def counts(self):
        with self.lock:
            return self.scan_count, self.odom_count


class TurtleBotGoalEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
        self,
        world_name: str = 'default',   # find yours: gz service -l | grep /control
        use_stamped: bool = True,      # Jazzy TurtleBot3 sim uses TwistStamped
        do_sim_reset: bool = True,     # reset Gazebo world at each episode start
        n_lidar: int = 24,
        lidar_max: float = 3.5,
        max_linear: float = 0.22,      # m/s  (TurtleBot3 Burger limit)
        max_angular: float = 1.5,      # rad/s
        step_dt: float = 0.1,          # seconds of (wall-clock) time per step
        max_steps: int = 400,
        goal_min: float = 0.8,         # goal distance range from start (m)
        goal_max: float = 1.6,
        goal_tol: float = 0.20,        # success radius (m)
        collision_dist: float = 0.16,  # lidar range counted as a collision (m)
    ):
        super().__init__()
        self.world_name = world_name
        self.do_sim_reset = do_sim_reset
        self.n_lidar = n_lidar
        self.lidar_max = lidar_max
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.step_dt = step_dt
        self.max_steps = max_steps
        self.goal_min = goal_min
        self.goal_max = goal_max
        self.goal_tol = goal_tol
        self.collision_dist = collision_dist

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_lidar + 4,), dtype=np.float32)

        # ROS setup (spun on a background thread so step() can just sleep)
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.ros = _RosIO(use_stamped)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.ros)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        self.goal = (0.0, 0.0)
        self.prev_dist = 0.0
        self.steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)

        self._wait_for_data(timeout=15.0, min_new=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wait_for_data(self, timeout: float = 5.0, min_new: int = 3):
        """Block until at least min_new fresh scan AND odom messages have arrived."""
        s0, o0 = self.ros.counts()
        t0 = time.time()
        while time.time() - t0 < timeout:
            s, o = self.ros.counts()
            if s - s0 >= min_new and o - o0 >= min_new:
                return
            time.sleep(0.02)
        raise RuntimeError(
            'Timed out waiting for /scan and /odom. Is Gazebo running, unpaused, '
            'and is the robot spawned? Check: ros2 topic hz /scan /odom')

    def _reset_sim(self):
        cmd = ['gz', 'service', '-s', f'/world/{self.world_name}/control',
               '--reqtype', 'gz.msgs.WorldControl', '--reptype', 'gz.msgs.Boolean',
               '--timeout', '3000', '--req', 'reset: {all: true}']
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode != 0 or 'false' in res.stdout.lower():
                self.ros.get_logger().warn(
                    f'Gazebo reset may have failed (world "{self.world_name}"): '
                    f'{res.stdout.strip()} {res.stderr.strip()}')
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.ros.get_logger().warn(f'Gazebo reset failed: {e}')
        time.sleep(0.5)

    def _lidar_bins(self, ranges: np.ndarray, range_min: float) -> np.ndarray:
        r = np.nan_to_num(ranges, nan=self.lidar_max,
                          posinf=self.lidar_max, neginf=range_min)
        r = np.clip(r, 0.0, self.lidar_max)
        return np.array([b.min() for b in np.array_split(r, self.n_lidar)],
                        dtype=np.float32)

    def _goal_polar(self, pose):
        x, y, yaw = pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - yaw)
        return dist, angle

    def _observe(self):
        scan, rmin, _, pose = self.ros.snapshot()
        bins = self._lidar_bins(scan, rmin)
        dist, angle = self._goal_polar(pose)
        obs = np.concatenate([
            bins / self.lidar_max,
            [min(dist / (2.0 * self.goal_max), 1.0), angle / math.pi],
            [self.last_action[0], self.last_action[1]],
        ]).astype(np.float32)
        obs[: self.n_lidar] = obs[: self.n_lidar] * 2.0 - 1.0   # map [0,1] -> [-1,1]
        return np.clip(obs, -1.0, 1.0), bins, dist, angle

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ros.publish_cmd(0.0, 0.0)
        if self.do_sim_reset:
            self._reset_sim()
        self._wait_for_data(timeout=10.0, min_new=3)

        # Sample goal in the robot's start frame, then express it in the odom frame
        _, _, _, pose = self.ros.snapshot()
        x0, y0, yaw0 = pose
        radius = float(self.np_random.uniform(self.goal_min, self.goal_max))
        bearing = float(self.np_random.uniform(-math.pi, math.pi))
        gx_l, gy_l = radius * math.cos(bearing), radius * math.sin(bearing)
        self.goal = (x0 + math.cos(yaw0) * gx_l - math.sin(yaw0) * gy_l,
                     y0 + math.sin(yaw0) * gx_l + math.cos(yaw0) * gy_l)

        self.steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        obs, _, dist, _ = self._observe()
        self.prev_dist = dist
        return obs, {'goal': self.goal, 'distance': dist}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        lin = (float(a[0]) + 1.0) / 2.0 * self.max_linear
        ang = float(a[1]) * self.max_angular
        self.ros.publish_cmd(lin, ang)
        self.last_action = a
        time.sleep(self.step_dt)
        self.steps += 1

        obs, bins, dist, angle = self._observe()
        min_range = float(bins.min())

        # ---- Reward ----
        reward = (self.prev_dist - dist) * 15.0     # progress toward goal
        reward -= 0.02                               # time penalty
        reward -= 0.05 * abs(a[1])                   # discourage spinning
        if min_range < 0.35:                         # discourage hugging obstacles
            reward -= (0.35 - min_range) * 2.0
        self.prev_dist = dist

        terminated = False
        truncated = False
        info = {'distance': dist, 'is_success': False}

        if dist < self.goal_tol:
            reward += 100.0
            terminated = True
            info['is_success'] = True
        elif min_range < self.collision_dist:
            reward -= 100.0
            terminated = True
            info['collision'] = True
        elif self.steps >= self.max_steps:
            truncated = True

        if terminated or truncated:
            self.ros.publish_cmd(0.0, 0.0)

        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ros.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        self._executor.shutdown()
        self.ros.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()