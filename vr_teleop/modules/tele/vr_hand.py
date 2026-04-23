from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pynput.keyboard import KeyCode, Listener
from sapien import Pose

from ...envs.utils.pose import PoseConfig
from ..retargeting.visualizer import HandVisualizer
from ..vr import VRHandViewer
from .base_tele import TeleOperatorBase, TeleOperatorConfig, teleoperator_builder
from .tele_flags import TeleFlags


if TYPE_CHECKING:
    from ..ik import SeqIKSolver
    from ..retargeting import Retargeting


_control_active = 0
_extern_record = 0
_done_flag = 0


def _on_key_press(key: KeyCode):
    if key == KeyCode.from_char("a"):
        global _extern_record
        _extern_record = 1

    if key == KeyCode.from_char("s"):
        global _control_active
        _control_active = (_control_active + 1) % 2

    if key == KeyCode.from_char("d"):
        global _done_flag
        _done_flag = 1


@dataclass
class VRHandTeleOperatorConfig(TeleOperatorConfig):
    handtrans: list[PoseConfig] = field(default_factory=lambda: [PoseConfig() for _ in range(2)])

    auto_sync: bool = True


@teleoperator_builder.register(VRHandTeleOperatorConfig, "vr_hand")
class VRHandTeleOperator(TeleOperatorBase):
    def __init__(
        self,
        config: VRHandTeleOperatorConfig,
        order: list[str],
        viewer: "VRHandViewer",
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

        assert self.retargeting is not None, "The retargeting should be provided"

        self.slices = slices

        self.init_actions = [
            np.zeros(len(self.slices[f"arm_{idx}"]) + len(self.slices[f"hand_{idx}"])) for idx in range(self.n_control)
        ]
        self.handtrans = [pose.to_sapien_pose() for pose in config.handtrans]

        self.auto_sync = config.auto_sync

        self.key_listener = Listener(on_press=_on_key_press)  # type: ignore
        self.key_listener.start()

    def __del__(self) -> None:
        if not self.auto_sync:
            self.key_listener.stop()

    def get_controller_pose(self) -> list[Pose] | None:
        """
        The pose are then pass to IK solver to solve robot qpos
        The pose are target ee pose for robots
        Return: poses = [Pose]
        """
        poses = []
        for idx in range(self.n_control):
            grasp_pose = self.viewer.hand_root_pose[self.indices[idx]]  # type: ignore
            if grasp_pose is None:  # pyright: ignore
                return None

            target_pose = grasp_pose
            poses.append(target_pose)

        return poses

    @staticmethod
    def identity(joint_pos: np.ndarray) -> bool:
        return bool(abs(np.linalg.norm(joint_pos[5] - joint_pos[0])) < 1e-3)

    def step(
        self,
        cur_qposs: list[np.ndarray],
        root_poses: list[Pose],
        # _hand_idx: list[int],
    ) -> tuple[TeleFlags, list[np.ndarray]]:  # , list[Pose]]:
        """
        cur_qposs: list[np.ndarray], list of current qpos of floating hands
        root_poses: list[Pose], list of current root poses of floating hands
        return:
            flags: list[TeleFlags], list of control status, here we only set flag = 1
            actions: list[np.ndarry], list of control actions, [(action_space, )]

        """
        flags = [TeleFlags.IK_SUCCESS for i in range(self.n_control)]

        poses = self.get_controller_pose()
        hand_poses = self.viewer.hand_pose  # type: ignore
        actions = [self.init_actions[idx].copy() for idx in range(self.n_control)]

        global _extern_record
        global _done_flag
        if _extern_record == 1:
            flags[0] = TeleFlags.RESET
            _extern_record = 0
        elif _done_flag == 1:
            flags[0] = TeleFlags.DONE
            _done_flag = 0
        else:
            global _control_active
            if poses is not None and (self.auto_sync or _control_active):  # onging teleop
                for idx in range(self.n_control):
                    if len(self.slices[f"arm_qpos_{idx}"]) != 0:
                        _status, qpos = self.ik_solver[idx].compute_IK(
                            [poses[idx]],  # type: ignore
                            cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]],
                            root_poses[idx],
                        )
                        if qpos is not None:
                            actions[idx][self.slices[f"arm_{idx}"]] = qpos.astype(np.float32)
                        else:
                            actions[idx][self.slices[f"arm_{idx}"]] = cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]]
                            flags[idx] = TeleFlags.IK_FAILED

                    hand_pose = hand_poses[self.indices[idx]]
                    if hand_pose is not None and not self.identity(hand_pose):
                        inv = (self.handtrans[self.indices[idx]] * poses[idx].inv()).to_transformation_matrix()
                        hand_pose = np.array(hand_pose) @ inv[:3, :3].T + inv[:3, 3]

                        hand_qpos = self.retargeting[idx].retarget(hand_pose)  # type: ignore
                        actions[idx][self.slices[f"hand_{idx}"]] = hand_qpos[self.slices[f"retargeting_{idx}"]].astype(
                            np.float32
                        )
                    else:
                        actions[idx][self.slices[f"hand_{idx}"]] = cur_qposs[idx][self.slices[f"hand_qpos_{idx}"]]
                        flags[idx] = TeleFlags.NO_HAND_POSE
            elif not (self.auto_sync or _control_active):
                for idx in range(self.n_control):
                    actions[idx][self.slices[f"arm_{idx}"]] = cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]]
                    actions[idx][self.slices[f"hand_{idx}"]] = cur_qposs[idx][self.slices[f"hand_qpos_{idx}"]]

                    flags[idx] = TeleFlags.NOT_IN_CONTROL
            else:
                for idx in range(self.n_control):
                    actions[idx][self.slices[f"arm_{idx}"]] = cur_qposs[idx][self.slices[f"arm_qpos_{idx}"]]
                    actions[idx][self.slices[f"hand_{idx}"]] = cur_qposs[idx][self.slices[f"hand_qpos_{idx}"]]

                    flags[idx] = TeleFlags.NO_HAND_POSE

            for idx in range(self.n_control):
                for key, value in self.delta_control[idx].items():
                    if value:
                        actions[idx][self.slices[f"{key}_{idx}"]] -= cur_qposs[idx][self.slices[f"{key}_qpos_{idx}"]]

                for key, value in self.normalize_action[idx].items():
                    if value:
                        actions[idx][self.slices[f"{key}_{idx}"]] = self._normalize_action(
                            actions[idx][self.slices[f"{key}_{idx}"]],
                            self.action_limits[idx][key],
                        )

        return flags[0], actions

    @property
    def action_components(self) -> dict[str, list[int]]:
        raise NotImplementedError

    def reset(self) -> None:
        if self.reset_state:
            global _control_active
            _control_active = 0
