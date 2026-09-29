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
        self._linear_x = 0.0
        self._angular_z = 0.0
        # Preallocated GPU action tensor
        self._action_buffer = torch.zeros((self.num_envs, 2), device=self.device)
        self.sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

    def _cmd_cb(self, msg: Twist):
        # Non-blocking CPU update; eliminates CUDA micro-kernel launches in ROS callback thread
        self._linear_x = float(msg.linear.x)
        self._angular_z = float(msg.angular.z)

    def get_action(self) -> torch.Tensor:
        # Synchronously update preallocated buffer without dynamic clone allocations
        self._action_buffer[:, 0] = self._linear_x
        self._action_buffer[:, 1] = self._angular_z
        return self._action_buffer
