from dataclasses import dataclass, field
from typing import cast

import numpy as np
import sapien
from loguru import logger
from sapien import Pose
from sapien.physx import PhysxArticulation
from sapien.wrapper.pinocchio_model import PinocchioModel

from ...configs.config import Config
from ...envs.utils.urdf_loader import URDF


from ...envs.utils.pose import PoseConfig


@dataclass
class IKConfig(Config):
    ee_pose_convertor: list[PoseConfig] = field(default_factory=lambda: [PoseConfig() for _ in range(2)])

    ee_name: list[str] | None = None
    fix_joint_indices: list[int] | None = None
    fix_joint_values: list[float] | None = None

    use_projected_ik: bool = False

    threshold: float = 1e-4
    n_retry: int = 0

    max_iterations: int = 4000

    weights: list[float] | None = None

    mod_2pi: bool = True

    low_pass_alpha: float = 0


pi2 = np.pi * 2
pi4 = np.pi * 4


class JointLimit:
    def __init__(self, robot: "PhysxArticulation", pmodel: "PinocchioModel", qmask: np.ndarray) -> None:
        super().__init__()

        is_revolute = []
        for j in robot.get_active_joints():
            assert len(j.limits) == 1, "Joint limits should be provided in the form of [[lower, upper]]"

            limits = j.limits[0]
            isinf = np.isinf(limits).all()

            is_revolute.append(j.type in ("revolute", "revolute_unwrapped"))

            if is_revolute[-1]:
                if j.type == "revolute_unwrapped":
                    assert not isinf, (
                        "we don't support infinite revolute joint limits for revolute_unwrapped joints. This is weird."
                    )
                assert limits[0] < limits[1], "Revolute joint limits should be in the form of [lower, upper]"

        assert qmask.dtype == np.bool_

        joint_limits = np.array([j.limits[0] for j in robot.get_active_joints()])
        joint_limits[np.logical_not(qmask)] = np.array([-np.inf, np.inf])
        self.lower = joint_limits[:, 0]
        self.upper = joint_limits[:, 1]

        self.is_revolute = np.array(is_revolute) & qmask
        pin_limits = []
        for idx, j in enumerate(robot.get_active_joints()):
            if j.type == "revolute":
                pin_limits.append([-np.inf, np.inf])
                pin_limits.append([-np.inf, np.inf])

            else:
                pin_limits.append(j.limits[0])

        pin_limits = np.array(pin_limits)
        self.pin_lower = pin_limits[:, 0]
        self.pin_higher = pin_limits[:, 1]

    def clip_pin_model(self, qpos: np.ndarray):
        return np.clip(qpos, self.pin_lower, self.pin_higher)

    def mod_2pi_(self, x: np.ndarray):
        x[self.is_revolute] = (x[self.is_revolute] + np.pi) % pi2 - np.pi
        return x

    def check_limit(self, qpos: np.ndarray):
        return np.all(qpos >= self.lower) and np.any(qpos <= self.upper)


class IKSolver:
    def __init__(self, robot: "PhysxArticulation", config: IKConfig) -> None:
        super().__init__()

        assert isinstance(robot, PhysxArticulation)
        self.use_projected_ik = config.use_projected_ik
        self.thresh = config.threshold
        self.max_iter = config.max_iterations
        self.n_try = config.n_retry + 1

        self.mod_2pi = config.mod_2pi

        assert config.ee_name is not None, "ee_name should be provided"

        ee_links = [robot.find_link_by_name(ee_name) for ee_name in config.ee_name]

        self.ee_link_idx = [robot.get_links().index(ee_link) for ee_link in ee_links]

        self.weights = None if config.weights is None else np.array(config.weights)

        self.qmask = np.ones(robot.dof, dtype=bool)
        if config.fix_joint_indices is not None:
            self.qmask[config.fix_joint_indices] = False
        if config.fix_joint_values is not None:
            assert config.fix_joint_indices is not None, "fix_joint_indices should be provided"
            assert len(config.fix_joint_indices) == len(config.fix_joint_values), (
                "fix_joint_indices and fix_joint_values should have the same length"
            )

            self.fixed_joint_indices = np.array(config.fix_joint_indices)
            self.fixed_joint_values = np.array(config.fix_joint_values)
        else:
            self.fixed_joint_values = None

        try:
            import pinocchio  # noqa: F401
        except ImportError:
            if config.use_projected_ik:
                logger.warning("Pinocchio is not installed, can not use projected_ik")
            config.use_projected_ik = False

        if config.use_projected_ik or self.weights is not None:
            from sapien.wrapper.pinocchio_model import _create_pinocchio_model

            from .projected_ik import compute_inverse_kinematics

            self.pmodel = _create_pinocchio_model(robot)
            self.compute_inverse_kinematics = compute_inverse_kinematics

        else:
            assert len(ee_links) == 1, "Only support one ee link when using default pinocchio"
            self.pmodel = cast("PinocchioModel", PhysxArticulation.create_pinocchio_model(robot))

        self.projector = JointLimit(robot, self.pmodel, self.qmask)
        self.lower = self.projector.lower.clip(-10000.0, 10000.0)
        self.upper = self.projector.upper.clip(-10000.0, 10000.0)

        self.ee_pose_convertor = [pose.to_sapien_pose() for pose in config.ee_pose_convertor]

    def compute_ee_pose(self, qpos: np.ndarray) -> list[Pose]:
        self.pmodel.compute_forward_kinematics(qpos)
        return [
            self.pmodel.get_link_pose(link_idx) * self.ee_pose_convertor[i].inv()
            for i, link_idx in enumerate(self.ee_link_idx)
        ]

    def projected_ik(self, goal_pose: tuple[Pose, ...], start_qpos: np.ndarray):
        """
        When using python pinocchio, we will activate the projected IK, which means that in every iteration of the
        gradient descent we will project the qpos to the joint limits.

        Otherwise, we only check joint limits after the optimization. In this case we will first mod
        the qpos to [-pi, pi] for revolute joints, and then check the joint limits.
        """
        index = self.ee_link_idx
        if self.use_projected_ik or self.weights is not None:
            qpos, sucess, err = self.compute_inverse_kinematics(
                self.pmodel,
                index,
                goal_pose,
                self.weights,
                start_qpos,
                self.qmask,
                self.thresh,
                self.max_iter,
                None,
            )
            status = "Success" if sucess else "IK Failed"

            return qpos, status, err
        else:
            assert len(goal_pose) == 1
            qpos, success, err = self.pmodel.compute_inverse_kinematics(
                index[0], goal_pose[0], start_qpos, self.qmask, max_iterations=self.max_iter, eps=self.thresh
            )

            if success:
                if not isinstance(err, np.float64):
                    err = np.linalg.norm(err)
                success = err < self.thresh
            # In this case qpos may out of the joint limits boundary for revolute joints
            if self.mod_2pi:
                qpos = self.projector.mod_2pi_(qpos)

            if success and self.projector and not self.projector.check_limit(qpos):
                return qpos, "Out joint limits", err
            status = "Success" if success else "IK Failed"
            return qpos, status, err

    def sampler(
        self,
        goal_pose: tuple[Pose, ...],
        start_qpos: np.ndarray,
        root_pose: Pose | None = None,
        is_absolute: bool = True,
    ):
        """
        Args:
            goal_pose: goal ee link pose, in sapein.Pose, note that the pose should be in the robot base frame
            start_qpos: current qpos
            root_pose: the root pose of the robot, in sapein.Pose
            is_absolute: bool, whether the goal pose is in the world frame or the robot base frame
            threshold: float, the threshold for the distance between the goal pose and the computed pose
        Returns:
            status: str, qpos: np.ndarry or None
            - "Success", qpos
            - "IK Failed", None
            - "Joint Limit Violation", None
        """
        if self.fixed_joint_values is not None:
            start_qpos[self.fixed_joint_indices] = self.fixed_joint_values

        if is_absolute:
            assert root_pose is not None, "root_pose should be provided"

            target_pose: tuple[Pose, ...] = tuple(
                root_pose.inv() * goal_pose[idx] * self.ee_pose_convertor[idx] for idx in range(len(goal_pose))
            )
        else:
            target_pose: tuple[Pose, ...] = tuple(
                goal_pose[idx] * self.ee_pose_convertor[idx] for idx in range(len(goal_pose))
            )

        for _ in range(self.n_try):
            yield self.projected_ik(target_pose, start_qpos.astype(np.float64))
            start_qpos[self.qmask] = np.random.uniform(self.lower, self.upper)[self.qmask]

    def compute_IK(
        self,
        goal_pose: tuple[Pose, ...],
        start_qpos: np.ndarray,
        root_pose: Pose | None = None,
        is_absolute: bool = True,
    ):
        for qpos, status, _err in self.sampler(goal_pose, start_qpos, root_pose, is_absolute):
            if status == "Success":
                return status, qpos

        return "Failed", None


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


class SeqIKSolver:
    def __init__(self, ik_solver: IKSolver, lp_filter: LPFilter | None = None):
        super().__init__()
        self.ik_solver = ik_solver
        self.lp_filter = lp_filter

    def compute_IK(
        self,
        goal_pose: tuple[Pose, ...],
        start_qpos: np.ndarray,
        root_pose: Pose | None = None,
        is_absolute: bool = True,
    ):
        status, qpos = self.ik_solver.compute_IK(goal_pose, start_qpos, root_pose, is_absolute)
        if status == "Success":
            if self.lp_filter is not None:
                qpos = self.lp_filter.next(qpos)
            return "Success", qpos
        return "Failed", None


@dataclass
class IKRobotConfig(Config):
    robot: URDF
    ik: IKConfig

    def create_ik_solver(self):
        scene = sapien.Scene()

        robot = self.robot.load(scene, "robot_for_ik")
        assert isinstance(robot, PhysxArticulation)

        iksolver = IKSolver(robot, self.ik)
        lp_filter = LPFilter(self.ik.low_pass_alpha) if self.ik.low_pass_alpha > 0 else None
        return SeqIKSolver(iksolver, lp_filter)
