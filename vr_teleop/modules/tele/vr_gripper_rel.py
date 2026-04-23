# from typing import List, Tuple


from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sapien import Pose


if TYPE_CHECKING:
    from ..ik import SeqIKSolver
    from ..retargeting import Retargeting

from ..vr import VRViewer
from .base_tele import TeleOperatorBase, TeleOperatorConfig, teleoperator_builder
from .tele_flags import TeleFlags


@dataclass
class VRGripperRelTeleOperatorConfig(TeleOperatorConfig):
    trigger_scale: float = 0.4
    mobile_speed: float = 0.1

    reset_state: bool = True
    reverse_gripper: bool = False  # whether to reverse the gripper control, i.e. trigger is open, and button is close
    trigger_limit: list[list[float]] | None = None

    change_viewpoint: bool = False
    move_wheel: bool = False
    clear_error: bool = False
    move_scale: float = 1.0
    reverse_rotation: bool = False


@teleoperator_builder.register(VRGripperRelTeleOperatorConfig, "vr_gripper_rel")
class VRGripperRelTeleOperator(TeleOperatorBase):
    def __init__(
        self,
        config: VRGripperRelTeleOperatorConfig,
        order: list[str],
        viewer: "VRViewer",
        slices: dict[str, list[int]],
        action_limits: list[np.ndarray],
        delta_control: list[dict[str, bool]],
        normalize_action: list[dict[str, bool]],
        reset_state: bool,
        ik_solver: "list[SeqIKSolver]",
        retargeting: "list[Retargeting] | None",
    ) -> None:
        super().__init__(
            config=config,
            order=order,
            viewer=viewer,
            slices=slices,
            action_limits=action_limits,
            delta_control=delta_control,
            normalize_action=normalize_action,
            reset_state=reset_state,
            ik_solver=ik_solver,
            retargeting=retargeting,
        )
        assert isinstance(self.viewer, VRViewer), "The viewer should be VRHandViewer"

        self.reset_state = config.reset_state
        self.in_control = [0 for _ in range(self.n_control)]

        self.reverse_gripper = config.reverse_gripper
        if config.trigger_limit is not None:
            self.trigger_limit = config.trigger_limit

        else:
            self.trigger_limit = [self.action_limits[idx]["gripper"][0] for idx in range(self.n_control)]

        if not self.reverse_gripper:
            self.trigger = [self.trigger_limit[idx][1] for idx in range(self.n_control)]
        else:
            self.trigger = [self.trigger_limit[idx][0] for idx in range(self.n_control)]

        self.trigger_scale = config.trigger_scale
        self.calibrate_pose = [None for _ in range(self.n_control)]
        self.calibrate_ee_pose = [None for _ in range(self.n_control)]

        self.prev_axes = [
            [0.0, 0.0] for _ in range(self.n_control if (not (config.change_viewpoint and config.move_wheel)) else 2)
        ]
        self.MOBILE_SPEED = config.mobile_speed

        self.prev_button_pressed = ["null" for _ in range(self.n_control)]

        self.init_actions = [
            np.zeros(
                len(self.slices.get(f"arm_{idx}", []))
                + len(self.slices.get(f"gripper_{idx}", []))
                + len(self.slices.get(f"body_{idx}", []))
                + len(self.slices.get(f"base_{idx}", []))
            )
            for idx in range(self.n_control)
        ]

    def get_control_state(self, cur_qposs: list[np.ndarray]) -> dict:
        flags = []

        for idx in range(self.n_control):
            button_pressed = self.viewer.pressed_button(self.indices[idx] + 1)  # type: ignore
            upper_button_state = self.viewer.up_button_state(self.indices[idx] + 1)  # type: ignore
            changed = button_pressed != self.prev_button_pressed[idx]

            trigger = (
                cur_qposs[idx][self.slices[f"gripper_{idx}"]] - (upper_button_state - 0.5) * self.trigger_scale * 2
            ) * (1 if not self.reverse_gripper else -1)

            self.trigger[idx] = np.clip(trigger, self.trigger_limit[idx][0], self.trigger_limit[idx][1])

            flag = TeleFlags.NOT_IN_CONTROL if self.in_control[idx] == 0 else TeleFlags.IK_SUCCESS

            if changed and (button_pressed == "down"):
                self.in_control[idx] = 1 - self.in_control[idx]
                if self.in_control[idx] == 0:
                    self.trigger[idx] = (
                        self.trigger_limit[idx][1] if not self.reverse_gripper else self.trigger_limit[idx][0]
                    )

            flag = TeleFlags.IK_SUCCESS if self.in_control[idx] == 1 else TeleFlags.NOT_IN_CONTROL

            if changed and (button_pressed == "B" or button_pressed == "both"):
                flag = TeleFlags.RESET

            if changed and (button_pressed == "A") and self.config.clear_error:
                flag = TeleFlags.CLEAR_ERROR

            elif changed and (button_pressed == "A") and (not self.config.clear_error):
                flag = TeleFlags.DONE

            self.prev_button_pressed[idx] = button_pressed  # type ignore

            flags.append(flag)

        states = {
            "flags": flags,
            "triggers": self.trigger,
        }
        return states

    def get_controller_pose(self) -> list[Pose]:
        """
        The pose are then pass to IK solver to solve robot qpos
        Return: poses = [Pose]
        """

        poses = []
        for idx in range(self.n_control):
            grasp_pose = self.viewer.controller_hand_poses[self.indices[idx]]  # type: ignore
            target_pose = grasp_pose
            poses.append(target_pose)

        return poses

    def get_controller_axes(self, idx: int) -> np.ndarray:
        """ ""
        The axes are used to control the robot's base / viewer root pose
        idx: i-th robot
        only support 2D control, i.e. x, y, rot
        return: list [ np.ndarray ], x, y, rot action
        """
        # TODO: too hacky
        indice = 1 - self.indices[0] if idx >= len(self.indices) else self.indices[idx]

        axes = self.viewer.controller_axes[indice]
        changed = self.diff(axes, self.prev_axes[idx])

        axes = self.clip_axes(axes) if changed else [0.0, 0.0]

        self.prev_axes[idx] = axes

        if abs(axes[1]) > abs(axes[0]):
            return np.array([axes[1] * self.MOBILE_SPEED, 0, 0])
        else:
            return np.array([0, 0, axes[0] * self.MOBILE_SPEED])

    @staticmethod
    def diff(a: list[float], b: list[float]) -> bool:
        return abs(a[0] - b[0]) > 0.1 or abs(a[1] - b[1]) > 0.1

    @staticmethod
    def clip_axes(axes: list[float]) -> list[float]:
        """
        clip the axes to: if abs(axes[i]) < 0.2, axies[i] = 0
        """
        if abs(axes[0]) < 0.2 and abs(axes[1]) < 0.2:
            axes = [0.0, 0.0]
        return axes

    def get_viewpoint_change_pose(self, idx: int) -> Pose:  #: , cur_qpos: np.ndarray) -> tuple[str, np.ndarray | None]:
        """
        a HACK for viewpoint change
        # use another controller to control the robot's base

        """
        control_axes = self.get_controller_axes(idx)
        root_movement = Pose(p=[control_axes[0], 0, 0], q=[np.cos(control_axes[1]), 0, 0, -np.sin(control_axes[1])])

        return root_movement

    def step(
        self,
        cur_qposs: list[np.ndarray],
        root_poses: list[Pose],
    ) -> tuple[TeleFlags, list[np.ndarray]]:
        states = self.get_control_state(cur_qposs)  # a dict of control states
        flags = states["flags"]
        poses = self.get_controller_pose()  # list of Pose

        actions = [self.init_actions[idx].copy() for idx in range(self.n_control)]

        if self.config.change_viewpoint:
            assert self.n_control == 1
            self.viewer.root_pose = self.viewer.root_pose * self.get_viewpoint_change_pose(
                1 if self.config.move_wheel else 0
            )

        for idx in range(self.n_control):
            if self.in_control[idx] == 1:
                if self.calibrate_pose[idx] is None:
                    self.calibrate_pose[idx] = (
                        root_poses[idx].inv() * poses[idx] * self.ik_solver[idx].ik_solver.ee_pose_convertor[0]
                    )

                    ee_pose = self.ik_solver[idx].ik_solver.compute_ee_pose(cur_qposs[idx][self.slices["arm_qpos_0"]])[
                        idx
                    ]
                    self.calibrate_ee_pose[idx] = ee_pose  # current ee pose

                new_pose = root_poses[idx].inv() * poses[idx] * self.ik_solver[idx].ik_solver.ee_pose_convertor[0]
                relative_pose_q = (Pose(q=new_pose.q) * Pose(q=self.calibrate_pose[idx].q).inv()).q
                relative_pose_p = new_pose.p - self.calibrate_pose[idx].p
                relative_pose = Pose(p=relative_pose_p, q=relative_pose_q)

                target_pose = Pose(q=relative_pose.q) * Pose(q=self.calibrate_ee_pose[idx].q)
                target_pose = Pose(p=relative_pose.p + self.calibrate_ee_pose[idx].p, q=target_pose.q)

                _status, qpos = self.ik_solver[idx].compute_IK(
                    [target_pose],  # type: ignore
                    cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]],
                    root_poses[idx],
                    is_absolute=False,
                )

                if qpos is not None:
                    actions[idx][self.slices[f"arm_{idx}"]] = qpos

                    if len(self.slices.get(f"body_qpos_{idx}", [])) != 0:
                        actions[idx][self.slices[f"body_{idx}"]] = qpos[self.slices[f"body_qpos_{idx}"]]

                else:
                    actions[idx][self.slices[f"arm_{idx}"]] = cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]]

                    if len(self.slices.get(f"body_qpos_{idx}", [])) != 0:
                        actions[idx][self.slices[f"body_{idx}"]] = cur_qposs[idx][self.slices[f"body_qpos_{idx}"]]

                    flags[idx] = TeleFlags.IK_FAILED

                if len(self.slices[f"gripper_qpos_{idx}"]) != 0:
                    actions[idx][self.slices[f"gripper_{idx}"]] = states["triggers"][idx]

                if self.config.move_wheel and len(self.slices.get(f"arm_qpos_{idx}", [])) > 0:
                    control_axes = self.get_controller_axes(idx)
                    actions[idx][self.slices[f"base_{idx}"]] = (
                        control_axes * self.config.move_scale * np.array([1, -1 if self.config.reverse_rotation else 1])
                    )

            else:
                actions[idx][self.slices[f"arm_{idx}"]] = cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]]

                if len(self.slices[f"gripper_qpos_{idx}"]) != 0:
                    actions[idx][self.slices[f"gripper_{idx}"]] = states["triggers"][idx]

        for idx in range(self.n_control):
            for key, value in self.delta_control[idx].items():
                if value:
                    actions[idx][self.slices[f"{key}_{idx}"]] -= cur_qposs[idx][self.slices[f"{key}_qpos_{idx}"]]

            for key, value in self.normalize_action[idx].items():
                if value:
                    if key == "base":
                        continue

                    actions[idx][self.slices[f"{key}_{idx}"]] = self._normalize_action(
                        actions[idx][self.slices[f"{key}_{idx}"]],
                        self.action_limits[idx][key],
                    )

        return states["flags"][0], actions

    def reset(self) -> None:  # , reset_state: bool = True) -> None:
        if self.reset_state:
            self.in_control = [0 for _ in range(self.n_control)]
        self.calibrate_pose = [None for _ in range(self.n_control)]
        self.calibrate_ee_pose = [None for _ in range(self.n_control)]

    @property
    def action_components(self) -> dict[str, list[int]]:
        raise NotImplementedError
