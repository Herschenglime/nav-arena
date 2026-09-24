import torch
from abc import ABC, abstractmethod
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist

class ActionAdapter(Node, ABC):
    """
    Base class for bridging asynchronous ROS 2 commands to synchronous Isaac Lab actions.
    Inherits from rclpy.Node to allow specialized subscription logic.
    """
    def __init__(self, node_name: str, num_envs: int, device: str):
        super().__init__(node_name)
        self.num_envs = num_envs
        self.device = device
        
        # Enforce sim time for the adapter so it uses /clock
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
    @abstractmethod
    def get_action(self) -> torch.Tensor:
        """
        Called synchronously by the main physics loop.
        Must return the action tensor for env.step().
        """
        pass

class TwistActionAdapter(ActionAdapter):
    """
    Adapter for differential drive platforms that consume geometry_msgs/Twist on /cmd_vel.
    """
    def __init__(self, num_envs: int, device: str):
        super().__init__('cmd_vel_adapter', num_envs, device)
        # Initialize buffer to zeros. Shape: [num_envs, 2] for [linear_x, angular_z]
        self._action_buffer = torch.zeros((self.num_envs, 2), device=self.device)
        self.sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

    def _cmd_cb(self, msg: Twist):
        # Asynchronously update the buffer with the latest command
        self._action_buffer[:, 0] = msg.linear.x
        self._action_buffer[:, 1] = msg.angular.z

    def get_action(self) -> torch.Tensor:
        # Synchronous read. Timeout logic is omitted for MVP.
        return self._action_buffer.clone()
