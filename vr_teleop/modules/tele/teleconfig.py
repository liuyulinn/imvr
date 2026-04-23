from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...configs.config import Config


from .base_tele import TeleOperatorBase

if TYPE_CHECKING:
    # from ...miniussd.maniskill.env_base import Env
    from ...envs.base_env import Env
    from ..ik import SeqIKSolver
    from ..retargeting import Retargeting
    from ..vr import VRHandViewer, VRViewer
    # try:
    #     from ussd.maniskill import ManiSkillRLEnv

    # except ImportError:
    #     from ...miniussd.maniskill.teleop_env import ManiSkillRLEnv


from sapien.utils import Viewer

# from ..real_robot import RealRobotEnv, RobotWrapper
# from ..recorder import RecordConfig
# from .keyboardconfig import KeyboardConfig
from .base_tele import TeleOperatorBuilderConfig, teleoperator_builder

# from .vrconfig import VRQuestConfig

# from digital_twin.modules.vr import VRHandViewer


# _order2hand = {0: "left", 1: "right"}
# _order2idx = {"left": 0, "right": 1}


@dataclass
class TeleConfig(Config):
    order: list[str]  # ["left", "right"]
    """
    which hand to control, 0 for left, 1 for right
    """

    config: TeleOperatorBuilderConfig
    # keyboardconfig: KeyboardConfig = KeyboardConfig.field()()

    base_action_joint_names: list[str] | None = None

    arm_action_joint_names: list[str] | None = None
    """
    arm action joint names for control, used for mimic joints
    """

    hand_action_joint_names: list[str] | None = None
    """
    hand action joint names for control, used for mimic joints
    """

    gripper_action_joint_names: list[str] | None = None
    """
    gripper action joint names for control, used for mimic joints
    """

    reset_state: bool = True
    """
    reset teleoperator state: in_control = 0
    """

    def create_tele_operator(
        self,
        viewer: "VRViewer | VRHandViewer | Viewer",
        retargeting: "list[Retargeting] | None",
        ik_solver: "list[SeqIKSolver]",
        env: "Env",
    ) -> "TeleOperatorBase":
        assert all(item in ("left", "right") for item in self.order), "List contains invalid items!"

        tele_type = self.config._type_

        slices, normalize_action, use_delta, joint_limits = self.get_slices(env, retargeting, teleop_type=tele_type)

        tele = teleoperator_builder.create(self.config._type_)(
            self.config.get_cfg(),
            order=self.order,
            viewer=viewer,
            slices=slices,
            action_limits=joint_limits,
            delta_control=use_delta,
            normalize_action=normalize_action,
            reset_state=self.reset_state,
            ik_solver=ik_solver,
            retargeting=retargeting,
        )

        return tele

    def get_slices(
        self, env: "Env", retargeting: "list[Retargeting] | None", teleop_type: str
    ) -> tuple[
        dict[str, list[int]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[np.ndarray],
    ]:
        """
        return:
            slices: dict[str, list[int]]
            normalize_action: dict[str, list[int]]
            use_delta: dict[str, list[int]]
            moving_part_control_names: list[str]
        """
        slices = {}
        normalize_action_list, use_delta_list, _joint_limits_list = env.agent_control_status

        joint_limits_list = [{} for _ in range(len(env._robots))]
        for idx, robot in enumerate(env._robots):
            if idx >= len(self.order):
                """
                    a HACK for the case that the number of robots in the environment is more than the number of teleoperators
                    i.e. we don't want to control the robot
                """
                break

            active_joint_names = env.agent_active_joint_names[idx]

            arm_action_joint_names = env.get_controlled_joint_names("arm", idx) + env.get_controlled_joint_names(
                "root", idx
            )

            joint_limits_list[idx]["arm"] = np.concatenate(
                [
                    _joint_limits_list[idx].get("arm", np.empty((0, 2), dtype=np.float32)),
                    _joint_limits_list[idx].get("root", np.empty((0, 2), dtype=np.float32)),
                ],
                axis=0,
            )
            if self.arm_action_joint_names is not None:
                arm_action_joint_names = self.arm_action_joint_names[idx]

            slices[f"arm_{idx}"] = [arm_action_joint_names.index(joint) for joint in arm_action_joint_names]
            slices[f"arm_qpos_{idx}"] = [active_joint_names.index(joint) for joint in arm_action_joint_names]

            if "gripper" in teleop_type:
                gripper_action_joint_names = env.get_controlled_joint_names(
                    "gripper", idx
                ) + env.get_controlled_joint_names("gripper_active", idx)
                joint_limits_list[idx]["gripper"] = np.concatenate(
                    [
                        _joint_limits_list[idx].get("gripper", np.empty((0, 2), dtype=np.float32)),
                        _joint_limits_list[idx].get("gripper_active", np.empty((0, 2), dtype=np.float32)),
                    ],
                    axis=0,
                )

                if self.gripper_action_joint_names is not None:
                    gripper_action_indice = [
                        gripper_action_joint_names.index(j_name) for j_name in self.gripper_action_joint_names[idx]
                    ]
                    joint_limits_list[idx]["gripper"] = joint_limits_list[idx]["gripper"][gripper_action_indice]
                    gripper_action_joint_names = self.gripper_action_joint_names[idx]

                if "gripper_body" in teleop_type:
                    for order_idx, order in enumerate(self.order):
                        slices[f"gripper_{order_idx}"] = [
                            gripper_action_joint_names.index(joint) + len(arm_action_joint_names)
                            for joint in gripper_action_joint_names
                            if order in joint
                        ]
                        slices[f"gripper_qpos_{order_idx}"] = [
                            active_joint_names.index(joint) for joint in gripper_action_joint_names if order in joint
                        ]

                else:
                    slices[f"gripper_{idx}"] = [
                        gripper_action_joint_names.index(joint) + len(arm_action_joint_names)
                        for joint in gripper_action_joint_names
                    ]
                    slices[f"gripper_qpos_{idx}"] = [
                        active_joint_names.index(joint) for joint in gripper_action_joint_names
                    ]

                    body_action_joint_names = env.get_controlled_joint_names("body", idx)
                    if len(body_action_joint_names):
                        joint_limits_list[idx]["body"] = _joint_limits_list[idx].get(
                            "body", np.empty((0, 2), dtype=np.float32)
                        )
                        slices[f"body_{idx}"] = [
                            body_action_joint_names.index(joint)
                            + len(arm_action_joint_names)
                            + len(gripper_action_joint_names)
                            for joint in body_action_joint_names
                        ]
                        slices[f"body_qpos_{idx}"] = [
                            active_joint_names.index(joint) for joint in body_action_joint_names
                        ]

                    if self.base_action_joint_names is None:
                        base_action_joint_names = env.get_controlled_joint_names("base", idx)
                    else:
                        base_action_joint_names = self.base_action_joint_names[idx]

                    if len(base_action_joint_names):
                        slices[f"base_{idx}"] = [
                            base_action_idx
                            + len(arm_action_joint_names)
                            + len(gripper_action_joint_names)
                            + len(body_action_joint_names)
                            for base_action_idx in [0, 1]
                        ]

                    slices[f"arm_ik_qpos_{idx}"] = sorted([
                        active_joint_names.index(joint)
                        for joint in base_action_joint_names + body_action_joint_names + arm_action_joint_names
                    ])

            elif "hand" in teleop_type:
                hand_action_joint_names = (
                    env.get_controlled_joint_names("hand", idx)
                    + env.get_controlled_joint_names("thumb", idx)
                    + env.get_controlled_joint_names("finger", idx)
                )
                joint_limits_list[idx]["hand"] = np.concatenate(
                    [
                        _joint_limits_list[idx].get("hand", np.empty((0, 2), dtype=np.float32)),
                        _joint_limits_list[idx].get("thumb", np.empty((0, 2), dtype=np.float32)),
                        _joint_limits_list[idx].get("finger", np.empty((0, 2), dtype=np.float32)),
                    ],
                    axis=0,
                )
                if self.hand_action_joint_names is not None:
                    hand_action_joint_names = self.hand_action_joint_names[idx]

                if "thumb" in normalize_action_list[idx]:
                    assert normalize_action_list[idx]["thumb"] == normalize_action_list[idx]["hand"]
                    assert use_delta_list[idx]["thumb"] == use_delta_list[idx]["hand"]

                    normalize_action_list[idx].pop("thumb")
                    use_delta_list[idx].pop("thumb")

                assert retargeting is not None

                if "hand_body" in teleop_type:
                    for order_idx, order in enumerate(self.order):
                        slices[f"hand_{order_idx}"] = [
                            hand_action_joint_names.index(joint) + len(arm_action_joint_names)
                            for joint in hand_action_joint_names
                            if order in joint
                        ]
                        slices[f"hand_qpos_{order_idx}"] = [
                            active_joint_names.index(joint) for joint in hand_action_joint_names if order in joint
                        ]

                        hand_control_names = retargeting[idx].retargeting.joint_names
                        hand_indices = [
                            hand_control_names.index(joint) for joint in hand_action_joint_names if order in joint
                        ]
                        slices[f"retargeting_{order_idx}"] = hand_indices

                else:
                    slices[f"hand_{idx}"] = [
                        hand_action_joint_names.index(joint) + len(arm_action_joint_names)
                        for joint in hand_action_joint_names
                    ]

                    slices[f"hand_qpos_{idx}"] = [active_joint_names.index(joint) for joint in hand_action_joint_names]

                    hand_control_names = retargeting[idx].retargeting.joint_names
                    hand_indices = [hand_control_names.index(joint) for joint in hand_action_joint_names]
                    slices[f"retargeting_{idx}"] = hand_indices

            else:
                raise ValueError(f"Unknown type {teleop_type}")

        for idx, action_dict in enumerate(normalize_action_list):
            for key in list(action_dict.keys()):
                if f"{key}_{idx}" not in slices:
                    print(
                        f"Warning: normalize_action_list[{idx}] has key {key}_{idx} but not in slices; this may cause issues."
                    )
                    if "gripper" in key:
                        value = action_dict[key]
                        del action_dict[key]
                        action_dict["gripper"] = value

        for idx, action_dict in enumerate(use_delta_list):
            for key in list(action_dict.keys()):
                if f"{key}_{idx}" not in slices:
                    print(
                        f"Warning: use_delta_list[{idx}] has key {key}_{idx} but not in slices; this may cause issues."
                    )
                    if "gripper" in key:
                        value = action_dict[key]
                        del action_dict[key]
                        action_dict["gripper"] = value

        return slices, normalize_action_list, use_delta_list, joint_limits_list
