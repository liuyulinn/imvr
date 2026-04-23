from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pynput.keyboard import KeyCode, Listener
from sapien import Pose
from sapien.utils import Viewer

from .tele_flags import TeleFlags

if TYPE_CHECKING:
    from ..ik import SeqIKSolver
    from ..retargeting import Retargeting

from dataclasses import field

from .base_tele import TeleOperatorBase, TeleOperatorConfig, teleoperator_builder

_move_dir = np.array([
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
])


pressed_key = None
call = 0


def _on_key_press(key: KeyCode):
    global call
    global pressed_key
    pressed_key = key
    call = 1

    print(f"key {key} pressed")


@dataclass
class KeyboardGripperTeleOperatorConfig(TeleOperatorConfig):
    move_keymap: list[str] = field(default_factory=lambda: ["h", "k", "u", "j", "n", "m"])
    """
    left, right, up, down, in, out
    """

    rotate_keymap: list[str] = field(default_factory=lambda: ["v", "b", "t", "y", "o", "p"])
    """
    left, right, up, down, in, out
    """

    control_keymap: str = "c"
    gripper_keymap: str = "x"
    done_keymap: str = "d"
    reset_keymap: str = "r"

    move_step: float = 0.03
    rotate_step: float = 0.03


@teleoperator_builder.register(KeyboardGripperTeleOperatorConfig, "keyboard_gripper")
class KeyboardGripperTeleOperator(TeleOperatorBase):
    def __init__(
        self,
        config: KeyboardGripperTeleOperatorConfig,
        order: list[str],
        viewer: "Viewer",
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
        assert isinstance(self.viewer, Viewer), "The viewer should be VRViewer"

        assert self.n_control == 1, "Only support one robot now"
        self.action_dof = len(self.slices["arm_0"]) + len(self.slices["gripper_0"])

        self.in_control = [0 for _ in range(self.n_control)]
        self.trigger = [1 for _ in range(self.n_control)]

        self.key_listener = Listener(on_press=_on_key_press)  # type: ignore
        self.key_listener.start()

        self.move_step = config.move_step
        self.move_key = config.move_keymap
        self.rotate_step = config.rotate_step
        self.rotate_key = config.rotate_keymap
        self.control_key = config.control_keymap
        self.gripper_key = config.gripper_keymap

        self.done_key = config.done_keymap
        self.reset_key = config.reset_keymap

    def __del__(self):
        self.key_listener.stop()

    def get_keyboard_pose(self, pose) -> "tuple[TeleFlags, Pose]":
        """
        Controls the robot end effector using keyboard.
        Returns: updated Pose
        """

        p, rpy = pose.p, pose.rpy

        move_step = self.move_step
        move_key = self.move_key

        global pressed_key
        global call

        flag = TeleFlags.NOT_IN_CONTROL if self.in_control[0] == 0 else TeleFlags.IK_SUCCESS

        if call == 1:
            call = 0

            for i in range(6):  # move control
                if pressed_key == KeyCode.from_char(move_key[i]):
                    p += move_step * _move_dir[i]

                pose.set_p(p)

            rotate_step = self.rotate_step
            rotate_key = self.rotate_key
            for i in range(6):
                if pressed_key == KeyCode.from_char(rotate_key[i]):
                    rpy += rotate_step * _move_dir[i]
                pose.set_rpy(rpy)

            control_key = self.control_key
            if pressed_key == KeyCode.from_char(control_key):
                self.in_control[0] = 1 - self.in_control[0]

            flag = TeleFlags.IK_SUCCESS if self.in_control[0] == 1 else TeleFlags.NOT_IN_CONTROL

            gripper_key = self.gripper_key

            if pressed_key == KeyCode.from_char(gripper_key):
                self.trigger[0] = 1 - self.trigger[0]

            done_key = self.done_key
            if pressed_key == KeyCode.from_char(done_key):
                flag = TeleFlags.DONE

            reset_key = self.reset_key
            if pressed_key == KeyCode.from_char(reset_key):
                flag = TeleFlags.RESET

        return flag, pose

    def step(
        self,
        cur_qposs: list[np.ndarray],
        root_poses: list[Pose],  # don't required
    ) -> tuple[list[TeleFlags], list[np.ndarray]]:
        # TODO: only support one robot now

        flag = TeleFlags.IK_SUCCESS
        cur_qpos = cur_qposs[0]
        root_pose = root_poses[0]
        ik_solver = self.ik_solver[0]
        ee_pose = ik_solver.ik_solver.compute_ee_pose(cur_qpos[self.slices["arm_qpos_0"]])[0]

        flag, target_pose = self.get_keyboard_pose(ee_pose)

        if self.in_control[0]:
            status, qpos = self.ik_solver[0].compute_IK(
                (target_pose,), cur_qpos[self.slices["arm_qpos_0"]], root_pose, is_absolute=False
            )

            action = cur_qpos[: self.action_dof].astype(np.float32)

            if qpos is not None:
                action[self.slices["arm_0"]] = qpos[: self.action_dof]
                action[self.slices["gripper_0"]] = self.trigger[0] * 0.85
                action = action.astype(np.float32)

            else:
                flag = TeleFlags.IK_FAILED
        else:
            action = cur_qpos[: self.action_dof].astype(np.float32)

        for key, value in self.delta_control[0].items():
            if value:
                action[self.slices[f"{key}_0"]] -= cur_qpos[self.slices[f"{key}_qpos_0"]].astype(np.float32)
        for key, value in self.normalize_action[0].items():
            if value:
                action[self.slices[f"{key}_0"]] = self._normalize_action(
                    action[self.slices[f"{key}_0"]], self.action_limits[0][self.slices[f"{key}_0"], ...]
                )

        return flag, [action]  # type: ignore

    def reset(self) -> None:
        self.recording = 0
        if self.reset_state:
            self.in_control = [0 for _ in range(self.n_control)]

    @property
    def action_components(self) -> dict[str, list[int]]:
        raise NotImplementedError
