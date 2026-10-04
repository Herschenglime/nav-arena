# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process adapters for iPlanner, ViNT, NavDP, VIPlanner, and X-NavDP.

Each class wraps the corresponding upstream agent, preserving its native preprocessing, history handling, strict
checkpoint loading, and stop semantics. Ported from the NavDP Isaac Sim integration (``isaac_policies.py``), which
dispatched on the planner name inside one class.

Import is cheap (NumPy only); torch and the upstream agents are imported when a policy is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..base import InProcessPolicy, InProcessPolicyCfg, Plan, PolicyObservation
from .checkpoints import default_checkpoint
from .loader import activate_planner
from .observation import as_path, prepare_observation


@dataclass
class IPlannerCfg(InProcessPolicyCfg):
    fear_threshold: float = 0.7
    """Predicted collision fear at or above which the policy requests a stop."""


@dataclass
class ViNTCfg(InProcessPolicyCfg):
    task: str = "imagegoal"


@dataclass
class NavDPCfg(InProcessPolicyCfg):
    stop_threshold: float = -3.0
    """Critic-value threshold below which NavDP's native recovery behavior triggers (repository eval default)."""


@dataclass
class VIPlannerCfg(InProcessPolicyCfg):
    fear_threshold: float = 0.7
    semantic_checkpoint: Path | None = None
    """Mask2Former checkpoint; defaults to ``viplanner/checkpoints/mask2former.pth``."""
    semantic_config: Path | None = None
    """Mask2Former config; defaults to MMDetection's COCO panoptic R50 config."""


@dataclass
class XNavDPCfg(InProcessPolicyCfg):
    robot_embodiment: int = 0
    """X-NavDP embodiment conditioning: 0 = Dingo, 1 = G1, 2 = Go2."""
    is_real: bool = False


_XNAVDP_EMBODIMENTS = {0: "dingo", 1: "g1", 2: "go2"}


class _NavDPFamilyPolicy(InProcessPolicy):
    """Shared construction: resolve config, seed RNGs, and load the upstream agent (or accept an injected one)."""

    cfg_cls: ClassVar[type[InProcessPolicyCfg]] = InProcessPolicyCfg

    def __init__(
        self,
        intrinsic: Any,
        cfg: InProcessPolicyCfg | None = None,
        *,
        agent: Any = None,
        **cfg_kwargs: Any,
    ) -> None:
        """
        Args:
            intrinsic: Camera intrinsic matrix ``[3, 3]``.
            cfg: Policy config; if omitted one is built from ``cfg_kwargs``.
            agent: Pre-built upstream agent (tests); skips checkpoint loading.
            **cfg_kwargs: Fields of this policy's config dataclass.
        """
        super().__init__(cfg if cfg is not None else self.cfg_cls(**cfg_kwargs))
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64)
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be [3, 3], got {self.intrinsic.shape}")
        if agent is not None:
            self.agent = agent
            return
        self.reseed(self.cfg.seed)
        self.agent = self._load_agent()

    def _checkpoint(self) -> Path:
        path = Path(self.cfg.checkpoint) if self.cfg.checkpoint is not None else default_checkpoint(self.name)
        if not path.is_file():
            raise FileNotFoundError(f"{self.name} checkpoint missing: {path}")
        return path

    def _load_agent(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def _require_point_goal(obs: PolicyObservation) -> np.ndarray:
        goal = None if obs.goal_body is None else np.asarray(obs.goal_body, dtype=np.float32)
        if goal is None or goal.shape != (1, 3) or not np.isfinite(goal).all():
            raise ValueError("Expected finite body-frame point goal [1,3]")
        return goal

    @staticmethod
    def _require_goal_image(obs: PolicyObservation) -> np.ndarray:
        image = obs.goal_image
        if image is None or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Image-goal planning requires an RGB goal image")
        return np.ascontiguousarray(image[None])


class IPlannerPolicy(_NavDPFamilyPolicy):
    """iPlanner: depth + relative point goal -> path and collision fear."""

    name = "iplanner"
    supported_tasks = ("pointgoal",)
    cfg_cls = IPlannerCfg

    def _load_agent(self) -> Any:
        folder = activate_planner(self.name)
        from iplanner_agent import IPlannerAgent

        return IPlannerAgent(
            self.intrinsic, str(self._checkpoint()), str(folder / "configs/iplanner.yaml"), device=self.cfg.device
        )

    def step(self, obs: PolicyObservation) -> Plan:
        # iPlanner masks invalid depth AFTER resizing; do not sanitize before.
        _, depths = prepare_observation(obs.rgb, obs.depth, sanitize_depth=False)
        goal = self._require_point_goal(obs)
        _, trajectory, fear = self.agent.step_pointgoal(depths, np.clip(goal, -5, 5))
        fear = float(fear[0].item())
        if not np.isfinite(fear):
            raise ValueError("iPlanner returned nonfinite fear")
        return Plan(as_path(trajectory), fear >= self.cfg.fear_threshold, {"fear": fear})


class ViNTPolicy(_NavDPFamilyPolicy):
    """ViNT: RGB history + goal image -> path. The point goal is never given to the model."""

    name = "vint"
    supported_tasks = ("imagegoal",)
    default_plan_hz = 3.0
    cfg_cls = ViNTCfg

    def _load_agent(self) -> Any:
        folder = activate_planner(self.name)
        from vint_agent import ViNTAgent

        agent = ViNTAgent(
            self.intrinsic,
            str(self._checkpoint()),
            str(folder / "configs/vint.yaml"),
            str(folder / "configs/robot_config.yaml"),
            device=self.cfg.device,
        )
        agent.reset(1)
        return agent

    def reset(self) -> None:
        self.agent.reset(1)

    def step(self, obs: PolicyObservation) -> Plan:
        images, _ = prepare_observation(obs.rgb, obs.depth)
        goal_images = self._require_goal_image(obs)
        # Preserve ViNT's native action scaling, history, interpolation, and distance-based stop mask.
        _, trajectory = self.agent.step_imagegoal(goal_images, images)
        path = as_path(trajectory)
        stopped = bool(np.all(path[:, :2] == 0))
        return Plan(path, stopped, {"native_zero_path": stopped})


class NavDPPolicy(_NavDPFamilyPolicy):
    """NavDP: RGB history + depth + point or image goal -> diffusion trajectory selected by a critic."""

    name = "navdp"
    supported_tasks = ("pointgoal", "imagegoal")
    cfg_cls = NavDPCfg

    def _load_agent(self) -> Any:
        import torch

        activate_planner(self.name)
        from policy_agent import NavDP_Agent
        from policy_network import NavDP_Policy

        # Reuse the original agent's preprocessing, memory and step methods. Its constructor passes the device into
        # the policy's 'channels' positional slot and loads with strict=False, so build the SAME network explicitly:
        # CPU / non-default CUDA devices work and missing weights cannot hide.
        agent = NavDP_Agent.__new__(NavDP_Agent)
        agent.image_intrinsic = self.intrinsic
        agent.device = self.cfg.device
        agent.predict_size, agent.image_size, agent.memory_size = 24, 224, 8
        agent.navi_former = NavDP_Policy(
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            device=self.cfg.device,
        )
        state = torch.load(self._checkpoint(), map_location="cpu", weights_only=True)
        # The pretraining checkpoint also holds auxiliary training heads absent from the inference network. Ignore ONLY
        # these known heads; strictly require every inference parameter.
        expected = agent.navi_former.state_dict()
        extras = set(state) - set(expected)
        auxiliary = {"pixel_aux_head.weight", "pixel_aux_head.bias", "image_aux_head.weight", "image_aux_head.bias"}
        if extras - auxiliary:
            raise ValueError(f"Unexpected NavDP checkpoint keys: {sorted(extras - auxiliary)}")
        state = {key: value for key, value in state.items() if key not in extras}
        agent.navi_former.load_state_dict(state, strict=True)
        agent.navi_former.to(self.cfg.device).eval()
        agent.reset(1, self.cfg.stop_threshold)
        return agent

    def reset(self) -> None:
        self.agent.reset(1, self.cfg.stop_threshold)

    def step(self, obs: PolicyObservation) -> Plan:
        images, depths = prepare_observation(obs.rgb, obs.depth)
        if self.cfg.task == "imagegoal":
            goal_images = self._require_goal_image(obs)
            trajectory, _, values, _ = self.agent.step_imagegoal(goal_images, images, depths)
        else:
            goal = self._require_point_goal(obs)
            trajectory, _, values, _ = self.agent.step_pointgoal(goal, images, depths)
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError("NavDP returned nonfinite critic values")
        return Plan(
            as_path(trajectory),
            diagnostics={
                "critic_max": float(values.max()),
                "critic_min": float(values.min()),
                "native_critic_recovery": bool(values.max() < self.cfg.stop_threshold),
            },
        )


class VIPlannerPolicy(_NavDPFamilyPolicy):
    """VIPlanner: depth + Mask2Former semantics predicted from RGB + point goal -> path and fear."""

    name = "viplanner"
    supported_tasks = ("pointgoal",)
    cfg_cls = VIPlannerCfg

    def _load_agent(self) -> Any:
        import torch

        folder = activate_planner(self.name)
        import mmdet
        from viplanner_agent import VIPlannerAgent

        semantic_checkpoint = self.cfg.semantic_checkpoint or default_checkpoint("viplanner_mask2former")
        semantic_config = self.cfg.semantic_config or (
            Path(mmdet.__file__).parent / ".mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
        )
        agent = VIPlannerAgent(
            self.intrinsic,
            str(semantic_checkpoint),
            str(semantic_config),
            str(self._checkpoint()),
            str(folder / "configs/viplanner.yaml"),
            device=self.cfg.device,
        )
        # MMDetection's convenience loader is non-strict: verify every segmentation parameter too, so a mismatched
        # model cannot run.
        semantic_state = torch.load(semantic_checkpoint, map_location="cpu", weights_only=True)
        agent.m2f_inference.model.load_state_dict(semantic_state["state_dict"], strict=True)
        agent.m2f_inference.debug = False
        return agent

    def step(self, obs: PolicyObservation) -> Plan:
        # Like iPlanner, VIPlanner masks invalid depth after resizing: pass raw depth.
        images, depths = prepare_observation(obs.rgb, obs.depth, sanitize_depth=False)
        goal = self._require_point_goal(obs)
        # MMDetection takes BGR arrays; the simulator and other policies use RGB.
        bgr = np.ascontiguousarray(images[..., ::-1])
        _, trajectory, fear = self.agent.step_pointgoal(bgr, depths, np.clip(goal, -10, 10))
        fear = float(fear[0].item())
        if not np.isfinite(fear):
            raise ValueError("VIPlanner returned nonfinite fear")
        return Plan(
            as_path(trajectory),
            fear >= self.cfg.fear_threshold,
            {"fear": fear, "semantic_source": "mask2former_rgb_prediction"},
        )


class XNavDPPolicy(_NavDPFamilyPolicy):
    """X-NavDP: RGB history + depth + point goal + robot pose, with embodiment conditioning (Dingo = 0)."""

    name = "x_navdp"
    supported_tasks = ("pointgoal",)
    requires_pose = True
    cfg_cls = XNavDPCfg

    def _load_agent(self) -> Any:
        import torch

        if self.cfg.robot_embodiment not in _XNAVDP_EMBODIMENTS:
            raise ValueError("X-NavDP embodiment must be 0 (Dingo), 1 (G1), or 2 (Go2)")
        activate_planner(self.name)
        from policy_agent import NavDP_Agent
        from policy_network_embodiment import NavDP_Policy_Embodiment

        agent = NavDP_Agent.__new__(NavDP_Agent)
        agent.image_intrinsic, agent.device = self.intrinsic, self.cfg.device
        agent.image_size, agent.memory_size, agent.predict_size = 224, 8, 24
        agent.embodiment, agent.is_real = self.cfg.robot_embodiment, bool(self.cfg.is_real)
        agent.navi_former = NavDP_Policy_Embodiment(temporal_depth=16, device=self.cfg.device)
        state = torch.load(self._checkpoint(), map_location="cpu", weights_only=True)
        expected = agent.navi_former.state_dict()
        extras = set(state) - set(expected)
        unused_prefixes = ("pixel_encoder.", "image_encoder.", "pixel_aux_head.", "image_aux_head.", "critic_head.")
        unknown = {key for key in extras if key != "log_alpha" and not key.startswith(unused_prefixes)}
        if unknown:
            raise ValueError(f"Unexpected X-NavDP checkpoint keys: {sorted(unknown)}")
        # The released PointNav evaluator omits the image/pixel goal branches and the pretraining critic. Require ALL
        # evaluator parameters.
        agent.navi_former.load_state_dict({k: v for k, v in state.items() if k not in extras}, strict=True)
        agent.navi_former.to(self.cfg.device).eval()
        agent.current_scene_name, agent._occ_cache = None, {}
        agent.save_counter = np.zeros(1000, dtype=np.int32)
        agent.save_plan_dir = None
        agent.reset(1)
        return agent

    def reset(self) -> None:
        self.agent.reset(1)

    def step(self, obs: PolicyObservation) -> Plan:
        images, depths = prepare_observation(obs.rgb, obs.depth)
        goal = self._require_point_goal(obs)
        position = None if obs.robot_position is None else np.asarray(obs.robot_position, dtype=np.float64)
        quaternion = None if obs.robot_quaternion is None else np.asarray(obs.robot_quaternion, dtype=np.float64)
        if (
            position is None
            or quaternion is None
            or position.shape != (1, 3)
            or quaternion.shape != (1, 4)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-3)
        ):
            raise ValueError("X-NavDP needs world position [1,3] and unit XYZW quaternion [1,4]")
        # Preserve native eight-sample Q selection, pose-based prefix guidance, and stuck handling. Gradients are
        # required by prefix guidance: do not wrap this call in inference_mode/no_grad.
        trajectory, _, values, _ = self.agent.step_pointgoal_with_guidance(goal, images, depths, position, quaternion)
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError("X-NavDP returned nonfinite Q values")
        return Plan(
            as_path(trajectory),
            diagnostics={
                "q_max": float(values.max()),
                "q_min": float(values.min()),
                "native_stuck": bool(self.agent.is_stuck[0]),
                "embodiment": _XNAVDP_EMBODIMENTS.get(self.agent.embodiment, "unknown"),
                "is_real": bool(self.agent.is_real),
            },
        )


ADAPTER_POLICIES: tuple[type[_NavDPFamilyPolicy], ...] = (
    IPlannerPolicy,
    ViNTPolicy,
    NavDPPolicy,
    VIPlannerPolicy,
    XNavDPPolicy,
)
"""Policies registered by default."""
