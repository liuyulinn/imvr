from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import nlopt
import numpy as np
import torch

if TYPE_CHECKING:
    from .robot_wrapper import RobotWrapper


class MimicAdaptor:
    def __init__(
        self,
        robot: "RobotWrapper",
        target_joint_names: list[str],
        source_joint_names: list[str],
        mimic_joint_names: list[str],
        multipliers: list[float],
        offsets: list[float],
    ):
        super().__init__()

        self.robot = robot
        self.target_joint_names = target_joint_names

        # Index mapping
        self.idx_pin2target = np.array([robot.get_joint_index(n) for n in target_joint_names])

        self.multipliers = np.array(multipliers)
        self.offsets = np.array(offsets)

        # Indices in the pinocchio
        self.idx_pin2source = np.array([robot.get_joint_index(name) for name in source_joint_names])
        self.idx_pin2mimic = np.array([robot.get_joint_index(name) for name in mimic_joint_names])

        # Indices in the output results
        self.idx_target2source = np.array([self.target_joint_names.index(n) for n in source_joint_names])

        # Dimension check
        len_source, len_mimic = (
            self.idx_target2source.shape[0],
            self.idx_pin2mimic.shape[0],
        )
        len_mul, len_offset = self.multipliers.shape[0], self.offsets.shape[0]
        if not (len_mimic == len_source == len_mul == len_offset):
            raise ValueError(
                f"Mimic joints setting dimension mismatch.\n"
                f"Source joints: {len_source}, mimic joints: {len_mimic}, multiplier: {len_mul}, offset: {len_offset}"
            )
        self.num_active_joints = len(robot.dof_joint_names) - len_mimic

        # Uniqueness check
        if len(mimic_joint_names) != len(np.unique(mimic_joint_names)):
            raise ValueError(f"Redundant mimic joint names: {mimic_joint_names}")

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        mimic_qpos = qpos[self.idx_pin2source] * self.multipliers + self.offsets
        qpos[self.idx_pin2mimic] = mimic_qpos
        return qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        target_jacobian = jacobian[..., self.idx_pin2target]
        mimic_joint_jacobian = jacobian[..., self.idx_pin2mimic] * self.multipliers

        for i, index in enumerate(self.idx_target2source):
            target_jacobian[..., index] += mimic_joint_jacobian[..., i]
        return target_jacobian


class Optimizer:
    def __init__(
        self,
        robot: "RobotWrapper",
        wrist_link_name: str,
        target_joint_names: list[str],
        adaptor: "MimicAdaptor | None" = None,
    ):
        super().__init__()

        self.robot = robot
        self.num_joints = robot.dof
        self.wrist_link_name = wrist_link_name
        self.wrist_link_id = robot.get_link_index(wrist_link_name)

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(f"Joint {target_joint_name} given does not appear to be in robot XML.")
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array([i for i in range(robot.dof) if i not in idx_pin2target], dtype=int)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        self.joint_limits = np.array(robot.joint_limits)
        assert not np.isinf(self.joint_limits).any(), "Joint limits should be finite"
        self.set_opt_joint_limit(self.joint_limits[self.idx_pin2target])

        # Kinematics adaptor
        self.adaptor = adaptor

    def set_opt_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: MimicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        fixed_idx = self.idx_pin2fixed
        mimic_idx = adaptor.idx_pin2mimic
        new_fixed_id = np.array([x for x in fixed_idx if x not in mimic_idx], dtype=int)
        self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        ref_value = ref_value.astype(np.float32)
        fixed_qpos = fixed_qpos.astype(np.float32)
        last_qpos = last_qpos.astype(np.float32)

        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )

        last_qpos = np.clip(last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])
        objective_fn = self.get_objective_function(ref_value, fixed_qpos, last_qpos)

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            output = np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            output = np.array(last_qpos, dtype=np.float32)

        robot_qpos = np.zeros(self.robot.dof)
        robot_qpos[self.idx_pin2fixed] = fixed_qpos
        robot_qpos[self.idx_pin2target] = output

        if self.adaptor is not None:
            robot_qpos = self.adaptor.forward_qpos(robot_qpos)
        return robot_qpos

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ) -> Callable: ...


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: "RobotWrapper",
        wrist_link_name: str,
        target_joint_names: list[str],
        target_origin_link_names: list[str],
        target_task_link_names: list[str],
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
        adaptor: "MimicAdaptor | None" = None,
    ):
        super().__init__(robot, wrist_link_name, target_joint_names, adaptor)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.origin_link_indices = torch.tensor([
            self.computed_link_names.index(name) for name in target_origin_link_names
        ])
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])

        # Cache link indices that will involve in kinematics computation

        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        target_vector = ref_value
        torch_target_vec = torch.as_tensor(target_vector)
        torch_target_vec.requires_grad_(False)

        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]  # type: ignore

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: "RobotWrapper",
        wrist_link_name: str,
        target_joint_names: list[str],
        target_task_link_names: list[str],
        huber_delta: float = 0.02,
        norm_delta: float = 4e-3,
        adaptor: "MimicAdaptor | None" = None,
    ) -> None:
        super().__init__(robot, wrist_link_name, target_joint_names, adaptor)

        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta
        self.computed_link_names = list(set(target_task_link_names))

        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        target_poses = ref_value
        torch_target_poses = torch.as_tensor(target_poses)
        torch_target_poses.requires_grad_(False)

        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            # body_poses = [self.robot.get_link_pose(index) for index in self.task_link_indices]
            target_link_poses = [self.robot.get_link_pose(index) for index in self.computed_link_indices]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            task_link_pos = torch_body_pos[self.task_link_indices, :]

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(task_link_pos, torch_target_poses)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]  # type: ignore

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]
            return result

        return objective
