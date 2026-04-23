import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
import sapien
from loguru import logger
from sapien.wrapper.pinocchio_model import _create_pinocchio_model

from ...configs.config import Config
from ...envs.utils.urdf_loader import URDF
from .optimizer import MimicAdaptor, PositionOptimizer, VectorOptimizer
from .robot_wrapper import RobotWrapper


class MimicJointRecord(TypedDict):
    source_names: list[str]
    mimic_joints: list[str]
    multipliers: list[float]
    offsets: list[float]


@dataclass
class RetargetingConfig(Config):
    """
    every pair of target_origin_link and target_task_link
    map to: target_link_human_indices
    """

    # The link on the robot hand which corresponding to the wrist of human hand
    wrist_link_name: str

    hand_model: URDF

    # Whether to add free joint to the root of the robot. Free joint enable the robot hand move freely in the space
    add_dummy_free_joint: bool = False

    # Source refers to the retargeting input, which usually corresponds to the human hand
    # The joint indices of human hand joint which corresponds to each link in the target_link_names
    type: str = "vector"

    target_link_human_indices: list[list[int]] | None = None

    # Vector retargeting link names
    target_joint_names: list[str] | None = None
    target_origin_link_names: list[str] | None = None
    target_task_link_names: list[str] | None = None

    # Scaling factor for vector retargeting only
    # For example, Allegro is 1.6 times larger than normal human hand, then this scaling factor should be 1.6
    scaling_factor: list[float] = field(default_factory=lambda: [1.0])

    # Optimization hyperparameter
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # Mimic joint tag
    ignore_mimic_joint: bool = False

    # Low pass filter
    low_pass_alpha: float = 0.1

    calibrate_qpos: list[float] | None = None

    def build(self):
        config = self
        if self.type == "vector":
            assert config.target_origin_link_names is not None
            assert config.target_task_link_names is not None
            assert len(config.target_origin_link_names) == len(config.target_task_link_names)
        else:
            assert config.target_task_link_names is not None
            assert config.target_link_human_indices is not None

        target_link_human_indices = np.array(self.target_link_human_indices)

        if self.type == "vector":
            assert target_link_human_indices.shape == (2, len(config.target_origin_link_names))
        else:
            assert target_link_human_indices.shape == (1, len(config.target_task_link_names))

        scene = sapien.Scene()

        mimic_joints, source_names, multipliers, offsets = [], [], [], []

        articulation_builder = config.hand_model.load_builder_after_editing(scene)

        for mimic_record in articulation_builder.mimic_joint_records:
            source_names.append(mimic_record.mimic)
            mimic_joints.append(mimic_record.joint)
            multipliers.append(mimic_record.multiplier)
            offsets.append(mimic_record.offset)

        sapien_robot = articulation_builder.build()
        assert isinstance(sapien_robot, sapien.physx.PhysxArticulation)
        pmodel = _create_pinocchio_model(sapien_robot)
        robot = RobotWrapper(pmodel, sapien_robot)

        self.joint_names = joint_names = list(
            config.target_joint_names if config.target_joint_names is not None else robot.dof_joint_names
        )
        if not config.ignore_mimic_joint and len(mimic_joints) > 0:
            adaptor = MimicAdaptor(robot, joint_names, source_names, mimic_joints, multipliers, offsets)
            logger.warning(
                "Mimic joint adaptor enabled. The mimic joint tags in the URDF will be considered during retargeting.\n"
                "To disable mimic joint adaptor, consider setting ignore_mimic_joint=True in the configuration."
            )
        else:
            adaptor = None

        if self.type == "vector":
            return VectorOptimizer(
                robot,
                config.wrist_link_name,
                joint_names,
                target_origin_link_names=config.target_origin_link_names,
                target_task_link_names=config.target_task_link_names,
                norm_delta=config.normal_delta,
                huber_delta=config.huber_delta,
                adaptor=adaptor,
            )
        else:
            return PositionOptimizer(
                robot,
                config.wrist_link_name,
                joint_names,
                target_task_link_names=config.target_task_link_names,
                norm_delta=config.normal_delta,
                huber_delta=config.huber_delta,
                adaptor=adaptor,
            )


"""
Below is the original implementation of the Retargeting class.
"""


class Retargeting:
    def __init__(self, config: RetargetingConfig) -> None:
        """
        Provide retargeting service for human hand keypoints to robot hand dof.
        everything is specified in the RetargetingConfig.
        """
        super().__init__()

        lp_filter = LPFilter(config.low_pass_alpha) if 0 <= config.low_pass_alpha <= 1 else None

        self.config = config

        self._indices = np.array(self.config.target_link_human_indices)
        if len(config.scaling_factor) == 1:
            config.scaling_factor = [config.scaling_factor[0]] * len(self._indices[0])
        assert len(config.scaling_factor) == len(self._indices[0])

        self.scaling = np.array(config.scaling_factor)[..., None]

        optimizer = config.build()

        self.retargeting = SeqRetargeting(optimizer, lp_filter=lp_filter)

    def retarget(self, joint_pos: np.ndarray) -> np.ndarray:
        """
        Retarget human hand keypoints to robot hand dof
        Args:
            joints_pos:
                human hand keypoints: (21, 3)
        Return:
            robot hand qpos: (dof,)
        """
        if isinstance(joint_pos, list):
            joint_pos = joint_pos[0]

        if self.config.type == "vector":
            origin_indices = self._indices[0, :]
            task_indices = self._indices[1, :]

            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        elif self.config.type == "position":
            ref_value = joint_pos[self._indices[0, :], :]
        return self.retargeting.retarget(ref_value * self.scaling)


class LPFilter:
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False


class SeqRetargeting:
    def __init__(self, optimizer: "VectorOptimizer | PositionOptimizer", lp_filter: "LPFilter | None" = None):
        super().__init__()
        self.optimizer = optimizer

        # Temporal information
        self.last_qpos = optimizer.joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(np.float32)
        self.accumulated_time = 0
        self.num_retargeting = 0

        # Filter
        self.filter = lp_filter

    def retarget(self, ref_value: np.ndarray, fixed_qpos: np.ndarray | None = None):
        if fixed_qpos is None:
            fixed_qpos = np.array([])
        tic = time.perf_counter()

        qpos = self.optimizer.retarget(ref_value, fixed_qpos, self.last_qpos)
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos
        robot_qpos = qpos.copy()

        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def verbose(self):
        min_value = self.optimizer.opt.last_optimum_value()
        print(f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s")
        print(f"Last distance: {min_value}")

    def reset(self):
        self.last_qpos = self.optimizer.joint_limits.mean(1)[self.optimizer.idx_pin2target]
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names
