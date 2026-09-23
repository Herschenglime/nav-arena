import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry


class DummyPolicyNode(Node):
    def __init__(self):
        super().__init__('dummy_policy_node')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.current_pose = None
        self.current_goal = None
        
        # Simple control loop timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Dummy Policy Node initialized.")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg: PoseStamped):
        self.current_goal = msg.pose
        self.get_logger().info(
            f"Received new goal: ({self.current_goal.position.x:.2f}, {self.current_goal.position.y:.2f})"
        )

    def euler_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        cmd = Twist()
        if self.current_pose is None or self.current_goal is None:
            self.cmd_vel_pub.publish(cmd)
            return

        curr_x = self.current_pose.position.x
        curr_y = self.current_pose.position.y
        curr_yaw = self.euler_from_quaternion(self.current_pose.orientation)

        goal_x = self.current_goal.position.x
        goal_y = self.current_goal.position.y

        dx = goal_x - curr_x
        dy = goal_y - curr_y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        
        yaw_error = target_yaw - curr_yaw
        while yaw_error > math.pi:
            yaw_error -= 2.0 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2.0 * math.pi

        # Thresholds match PointNavTask
        if distance > 0.4:
            if abs(yaw_error) > 0.2:
                cmd.angular.z = max(-1.0, min(1.0, 1.0 * yaw_error))
            else:
                cmd.linear.x = max(0.0, min(1.0, 1.0 * distance))
                cmd.angular.z = max(-1.0, min(1.0, 0.5 * yaw_error))
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DummyPolicyNode()
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
