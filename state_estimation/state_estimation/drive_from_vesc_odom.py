"""Recover the steering command from /vesc/odom and republish it on /drive.

Workaround for bags recorded without /drive (e.g. the June 2026 NUC6 bags).
The EKF needs the steering angle as a model input. On NUC6, vesc_to_odom runs
with use_servo_cmd_to_calc_angular_velocity: true, so the yaw rate in
/vesc/odom is NOT measured: it is computed from the servo command as

    omega = v * tan(delta_cmd) / L        (L = wheelbase in NUC6/vesc.yaml)

Inverting gives the commanded steering angle:

    delta_cmd = atan(L * omega / v)

Notes:
- This is the COMMANDED steering, not the true wheel angle. The full model
  applies its own steer_lag filter on top.
- It does not use the IMU, so the gyro is not counted twice in the EKF.
- Undefined near standstill: below v_min the last value is held.
- The sign is NOT flipped here. The EKF flips it itself
  (ackermann_data[1] = -msg.drive.steering_angle), as for a real /drive.
- drive.speed is the measured VESC speed. The old single_track_model
  differentiates it to get acceleration, which will be noisy.

Usage (bag replay):
    ros2 run state_estimation drive_from_vesc_odom --ros-args -p use_sim_time:=true
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class DriveFromVescOdom(Node):

    def __init__(self):
        super().__init__('drive_from_vesc_odom')
        # Must match vesc_to_odom's wheelbase in stack_master/config/NUC6/vesc.yaml
        # (0.321), NOT car_model.yaml (0.335): we invert vesc_to_odom's own formula.
        self.L = self.declare_parameter('wheelbase', 0.321).value
        self.v_min = self.declare_parameter('v_min', 0.3).value  # [m/s]
        in_topic = self.declare_parameter('odom_topic', '/vesc/odom').value
        out_topic = self.declare_parameter('drive_topic', '/drive').value

        self.steer = 0.0
        self.pub = self.create_publisher(AckermannDriveStamped, out_topic, 10)
        self.create_subscription(Odometry, in_topic, self.odom_callback, 10)
        self.get_logger().info(
            f'Republishing {in_topic} -> {out_topic} (L={self.L} m, v_min={self.v_min} m/s)')

    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        if abs(v) > self.v_min:
            self.steer = math.atan(self.L * omega / v)

        out = AckermannDriveStamped()
        out.header = msg.header  # keep the bag stamp: the EKF uses it for dt
        out.drive.steering_angle = self.steer
        out.drive.speed = v
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DriveFromVescOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
