import rclpy
import pytest
import torch
import time
from geometry_msgs.msg import Twist
from nav_arena.ros2.adapters.action_adapter import TwistActionAdapter

def test_twist_action_adapter():
    rclpy.init()
    try:
        # Create the adapter with 2 environments
        adapter = TwistActionAdapter(num_envs=2, device='cpu')
        
        # Verify initial action is zero
        initial_action = adapter.get_action()
        assert torch.all(initial_action == 0.0)
        assert initial_action.shape == (2, 2)
        
        # Create a mock publisher
        node = rclpy.create_node('mock_publisher')
        pub = node.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publish a command
        msg = Twist()
        msg.linear.x = 1.5
        msg.angular.z = -0.5
        pub.publish(msg)
        
        # Spin to let the adapter process the message
        # We might need a small delay and a few spins to ensure delivery
        start_time = time.time()
        msg_received = False
        while time.time() - start_time < 2.0:
            rclpy.spin_once(node, timeout_sec=0.1)
            rclpy.spin_once(adapter, timeout_sec=0.1)
            action = adapter.get_action()
            if action[0, 0].item() == 1.5 and action[0, 1].item() == -0.5:
                msg_received = True
                break
                
        assert msg_received, "Adapter did not receive the Twist message within the timeout."
        
        # Verify both environments get the same command (since it's broadcasted)
        assert action[1, 0].item() == 1.5
        assert action[1, 1].item() == -0.5
        
    finally:
        node.destroy_node()
        adapter.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    test_twist_action_adapter()
    print("test_twist_action_adapter PASSED")
