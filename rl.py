#!/usr/bin/env python3
"""
Basic template: two-wheel differential drive robot with a 2D lidar (ROS 2).

- Subscribes : /scan     (sensor_msgs/LaserScan)
- Publishes  : /cmd_vel  (geometry_msgs/Twist)

Behaviour: drive forward, steer away from nearby obstacles,
turn in place toward the freer side if blocked.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import LaserScan


class DiffDriveLidar(Node):
    def __init__(self):
        super().__init__('diff_drive_lidar')

        # ---- Parameters (tune for your robot) ----
        self.declare_parameter('max_linear', 0.20)     # m/s
        self.declare_parameter('max_angular', 1.0)     # rad/s
        self.declare_parameter('stop_dist', 0.35)      # m, blocked if closer
        self.declare_parameter('slow_dist', 0.80)      # m, start slowing/steering
        self.declare_parameter('wheel_base', 0.16)     # m, distance between wheels
        self.declare_parameter('wheel_radius', 0.033)  # m

        # Jazzy TurtleBot3 sim expects TwistStamped; set False for plain Twist
        self.declare_parameter('use_stamped', True)
        self.use_stamped = self.get_parameter('use_stamped').value
        msg_type = TwistStamped if self.use_stamped else Twist
        self.cmd_pub = self.create_publisher(msg_type, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

        self.get_logger().info('Differential drive lidar node started')

    # ------------------------------------------------------------------
    # Lidar helpers
    # ------------------------------------------------------------------
    @staticmethod
    def sector_min(scan: LaserScan, start_deg: float, end_deg: float) -> float:
        """Minimum valid range between two angles (deg). 0 deg = forward, +ve = left."""
        start, end = math.radians(start_deg), math.radians(end_deg)
        best = scan.range_max
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))  # wrap to [-pi, pi]
            if start <= angle <= end:
                best = min(best, r)
        return best

    # ------------------------------------------------------------------
    # Main control logic (runs on every scan)
    # ------------------------------------------------------------------
    def scan_cb(self, scan: LaserScan):
        max_lin = self.get_parameter('max_linear').value
        max_ang = self.get_parameter('max_angular').value
        stop_d = self.get_parameter('stop_dist').value
        slow_d = self.get_parameter('slow_dist').value

        front = self.sector_min(scan, -30, 30)
        left = self.sector_min(scan, 30, 90)
        right = self.sector_min(scan, -90, -30)

        cmd = Twist()

        if front < stop_d:
            # Blocked: stop and rotate toward the side with more space
            cmd.linear.x = 0.0
            cmd.angular.z = max_ang if left > right else -max_ang
        elif front < slow_d:
            # Obstacle ahead: slow down and steer away
            scale = (front - stop_d) / (slow_d - stop_d)
            cmd.linear.x = max_lin * scale
            cmd.angular.z = 0.6 * max_ang if left > right else -0.6 * max_ang
        else:
            # Clear: go forward, gently stay away from side walls
            cmd.linear.x = max_lin
            cmd.angular.z = 0.5 * (left - right) / max(left + right, 1e-3)

        self.publish_cmd(cmd)

    def publish_cmd(self, twist: Twist):
        """Publish as Twist or TwistStamped depending on the use_stamped parameter."""
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist = twist
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(twist)

    # ------------------------------------------------------------------
    # Differential drive kinematics (use if driving motors directly)
    # ------------------------------------------------------------------
    def twist_to_wheels(self, v: float, w: float):
        """Convert body velocity (v m/s, w rad/s) to wheel angular speeds (rad/s)."""
        L = self.get_parameter('wheel_base').value
        R = self.get_parameter('wheel_radius').value
        v_left = v - w * L / 2.0
        v_right = v + w * L / 2.0
        return v_left / R, v_right / R

    def stop(self):
        self.publish_cmd(Twist())


def main():
    rclpy.init()
    node = DiffDriveLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()