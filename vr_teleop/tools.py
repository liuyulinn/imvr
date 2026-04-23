#!/usr/bin/env python3
import tyro
from typing import Annotated, Optional
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
import tqdm
import yaml
from sapien import Pose
import subprocess

from vr_teleop.configs.config import Config
from vr_teleop.envs import EnvBuilderConfig, env_builder
from vr_teleop.envs.utils import tree_utils as tu


@dataclass
class TaskConfig(Config):
    def run(self, path: Path):
        raise NotImplementedError


class ConfigBuilder:
    def __init__(self):
        self.default_configs = {}

    def register(self, name: str):
        def wrapper(cls):
            self.default_configs[name] = cls
            return cls

        return wrapper


task_builder = ConfigBuilder()


@dataclass
@task_builder.register("view")
class view(TaskConfig):
    """
    Visualize the environment
    """

    paused: bool = False
    random_action: bool = False

    render: bool = True

    hz: int | None = None

    def run(self, path: Path):
        inp = EnvBuilderConfig.from_yaml(str(path))
        env_config = inp.get_cfg()
        if inp._type_ == "maniskill_env":
            env_config.render_mode = "human"  # a HACK
        env = env_builder.create(inp._type_)(1, env_config)

        env.reset()

        env.viewer.paused = self.paused
        if self.render:
            env.render()

        s = time.time()
        while not env.viewer.closed:
            if self.hz is not None:
                s = time.time()
            if self.random_action:
                random_action = env.sample_random_action()
                env.step([random_action])

            if self.render:
                env.render()
                if self.hz is not None:
                    time.sleep(max(1.0 / self.hz - (time.time() - s), 0.0))


@dataclass
@task_builder.register("replay")
class replay(TaskConfig):
    """
    For replay/visualize trajectory
    """

    # args copied from https://github.com/haosulab/ManiSkill/blob/main/mani_skill/trajectory/replay_trajectory.py

    # traj_path: Annotated[Path, tyro.conf.arg(aliases=["-p"])] = Path("trajectory.h5")
    # """Path to the trajectory .h5 file to replay"""
    # traj_path is the input path to run function

    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion. Furthermore control mode conversion is not supported in GPU parallelized environments.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = True
    """Whether to save videos"""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data based on the environment's evaluation returned "success" label"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = True
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras. Default is none meaning it will use whatever was used in the original data collection or the environment default.
    Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""

    def run(self, path: Path):
        cmd = [sys.executable, "-m", "mani_skill.trajectory.replay_trajectory"]

        # Build command with all arguments
        cmd.extend(["--traj-path", str(path)])

        if self.sim_backend:
            cmd.extend(["--sim-backend", self.sim_backend])
        if self.obs_mode:
            cmd.extend(["--obs-mode", self.obs_mode])
        if self.target_control_mode:
            cmd.extend(["--target-control-mode", self.target_control_mode])
        if self.count:
            cmd.extend(["--count", str(self.count)])
        if self.reward_mode:
            cmd.extend(["--reward-mode", self.reward_mode])
        if self.shader:
            cmd.extend(["--shader", self.shader])
        if self.video_fps:
            cmd.extend(["--video-fps", str(self.video_fps)])

        cmd.extend(["--render-mode", self.render_mode])
        cmd.extend(["--num-envs", str(self.num_envs)])
        cmd.extend(["--max-retry", str(self.max_retry)])

        # Boolean flags
        if self.verbose:
            cmd.append("--verbose")
        if self.save_traj:
            cmd.append("--save-traj")
        if self.save_video:
            cmd.append("--save-video")
        if self.discard_timeout:
            cmd.append("--discard-timeout")
        if self.allow_failure:
            cmd.append("--allow-failure")
        if self.vis:
            cmd.append("--vis")
        if self.use_env_states:
            cmd.append("--use-env-states")
        if self.use_first_env_state:
            cmd.append("--use-first-env-state")
        if self.record_rewards:
            cmd.append("--record-rewards")

        # Execute
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("Replay completed!")
        except subprocess.CalledProcessError as e:
            print(f"Replay failed: {e}")
            raise


@dataclass
@task_builder.register("calibrate")
class calibrate(TaskConfig):
    """
    For calibrate and auto compute the scaling factor for retargeting

    it will launch VR hand viewer for 30s to capture the keypoints
    """

    _order2idx = {"left": 0, "right": 1}

    @staticmethod
    def identity(joint_pos: np.ndarray) -> bool:
        return bool(abs(np.linalg.norm(joint_pos[5] - joint_pos[0])) < 1e-3)

    @staticmethod
    def get_controlled_joint_names(name: str, robot):  #: AgentEntity):
        all_joints = robot.articulation.get_active_joints()
        controll_joints = robot.controller.controllers.get(name, None)
        if not controll_joints:
            return []

        return [all_joints[i].name for i in controll_joints.active_joint_indices]

    finger_name: list[str] = field(default_factory=lambda: ["thumb", "index", "middle", "ring", "pinky"])

    @staticmethod
    def find_link_by_name(name: str, robot):  #: AgentEntity):
        for link in robot.articulation.get_links():
            if link.name == name:
                return link
        raise ValueError(f"Link {name} not found")

    @staticmethod
    def get_link_pos(link):
        return link.pose.p[0]

    @staticmethod
    def get_link_pose(link):
        return Pose(p=link.pose.p[0], q=link.pose.q[0])

    def run(self, path: Path):
        import sapien

        from vr_teleop.modules.retargeting import Retargeting
        from vr_teleop.modules.teleconfig import TeleopConfig
        from vr_teleop.modules.vr import VRViewer, VRHandViewer

        scene = sapien.Scene()

        teleconfig = TeleopConfig.from_yaml(path)

        robot_scale_list = []
        target_link_human_indices_list = []
        tele_order_indices = [self._order2idx[order] for order in teleconfig.tele.order]

        for idx in range(len(tele_order_indices)):
            # order_idx = teleconfig.vr.vrconfig.order[idx]
            retarget_config = teleconfig.retargeting if idx == 0 else teleconfig.retargeting2
            assert retarget_config is not None

            target_link_human_indices = np.array(retarget_config.target_link_human_indices)
            target_task_link_names = retarget_config.target_task_link_names
            target_origin_link_names = retarget_config.target_origin_link_names

            assert (
                target_link_human_indices is not None
                and target_task_link_names is not None
                and target_origin_link_names is not None
            )

            target_link_human_indices_list.append(target_link_human_indices)

            retargeting = Retargeting(retarget_config)

            # reuse some code from retargeting.py
            optimizer = retargeting.retargeting.optimizer
            print("optimizer qpos names", optimizer.robot.dof_joint_names)

            qpos = np.array(retarget_config.calibrate_qpos)
            if optimizer.adaptor is not None:
                qpos[:] = optimizer.adaptor.forward_qpos(qpos)

            optimizer.robot.compute_forward_kinematics(qpos)
            target_link_poses = [optimizer.robot.get_link_pose(index) for index in optimizer.computed_link_indices]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            origin_link_pos = body_pos[optimizer.origin_link_indices, :]
            task_link_pos = body_pos[optimizer.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            robot_hand_scale = np.linalg.norm(robot_vec, axis=1)

            # store important information

            robot_scale_list.append(robot_hand_scale)
            print(f"robot_hand_size {idx}: ", robot_hand_scale)

        human_scale_list = [
            np.zeros(len(target_link_human_indice[0])) for target_link_human_indice in target_link_human_indices_list
        ]

        print("Launch VR hand viewer for 15s to capture the keypoints")
        vrviewer = VRHandViewer(visualize=True)
        vrviewer.set_scene(scene)

        while True:
            cur_time = time.time()
            count = [0] * len(tele_order_indices)
            while time.time() - cur_time < 15:
                for idx in range(len(tele_order_indices)):
                    human_hand_pos: np.ndarray | None = vrviewer.hand_pose[tele_order_indices[idx]]  # (21, 3)
                    if human_hand_pos is not None and not self.identity(human_hand_pos):
                        human_scale_list[idx] += np.linalg.norm(
                            human_hand_pos[target_link_human_indices_list[idx][1, ...]]
                            - human_hand_pos[target_link_human_indices_list[idx][0, ...]],
                            axis=1,
                        )
                        count[idx] += 1
                vrviewer.render()

            if 0 in count:
                print("No keypoints detected! Please try again.")
                continue

            human_scale_list = [human_scale_list[idx] / count[idx] for idx in range(len(tele_order_indices))]

            for idx in range(len(tele_order_indices)):
                print(f"human_hand_size: {idx}", human_scale_list[idx])
                print(f"retargeting scale: {idx}", (robot_scale_list[idx] / human_scale_list[idx]).tolist())
            break
        del vrviewer


@dataclass
@task_builder.register("convert")
class convert(TaskConfig):
    """
    compute IK convention convertor
    """

    axis_mapping = {
        "x": np.array([1, 0, 0]),
        "-x": np.array([-1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "-y": np.array([0, -1, 0]),
        "z": np.array([0, 0, 1]),
        "-z": np.array([0, 0, -1]),
    }

    def parse_axes_order(self, axes_string):
        """Parse axes order from string like 'x -y -z' or '-x z y'."""
        # Split by spaces and filter out empty strings
        axes = [axis.strip() for axis in axes_string.split() if axis.strip()]

        if len(axes) != 3:
            raise ValueError(f"Expected exactly 3 axes, got {len(axes)}: {axes}")

        # Validate that all axes are in our mapping
        for axis in axes:
            if axis not in self.axis_mapping:
                valid_axes = list(self.axis_mapping.keys())
                raise ValueError(f"Invalid axis '{axis}'. Valid axes are: {valid_axes}")

        return tuple(axes)

    def rotation_from_axes_order(self, order):
        """Construct rotation matrix from axis permutation order like ('z', '-y', 'x')."""
        if len(order) != 3:
            raise ValueError("Order must contain exactly 3 axes.")

        # Get the vectors corresponding to the order
        col1 = self.axis_mapping[order[0]]  # First column of the rotation matrix
        col2 = self.axis_mapping[order[1]]  # Second column of the rotation matrix
        col3 = self.axis_mapping[order[2]]  # Third column of the rotation matrix

        # Ensure that the basis is orthonormal (right-hand rule)
        rotation_matrix = np.column_stack((col1, col2, col3))

        # Check if the matrix is orthogonal and has determinant 1 (valid rotation)
        if not np.allclose(np.linalg.det(rotation_matrix), 1):
            raise ValueError("Invalid axis permutation: resulting matrix is not a proper rotation matrix.")

        return rotation_matrix

    def pose_from_axes_order(self, order):
        rotation = self.rotation_from_axes_order(order)
        rotation = np.concatenate(
            [
                np.concatenate([rotation, np.array([0, 0, 0]).reshape(3, 1)], axis=1),
                np.array([0, 0, 0, 1]).reshape(1, 4),
            ],
            axis=0,
        )

        pose = Pose(rotation)
        print(pose)
        print(pose.to_transformation_matrix())

    def run(self, path: str):
        """
        Run the conversion with axes order specified as a string.

        Args:
            path: String containing space-separated axes like "x -y -z" or "-x z y"
        """
        path = str(path)  # a HACK to ensure input is a string

        try:
            axes_order = self.parse_axes_order(path)
            print(f"Parsed axes order: {axes_order}")
            self.pose_from_axes_order(axes_order)
        except ValueError as e:
            print(f"Error parsing axes: {e}")
            print('Usage example: python tools.py run "x -y -z"')
            print("Valid axes: x, -x, y, -y, z, -z")


@dataclass
@task_builder.register("vis_ik")
class vis_ik(TaskConfig):
    """
    visualize robot for IK solver
    """

    def run(self, path: Path):
        import sapien

        from vr_teleop.modules.teleconfig import TeleopConfig

        scene = sapien.Scene()

        teleconfig = TeleopConfig.from_yaml(path)

        scene = sapien.Scene()
        scene.set_ambient_light(np.array([0.6, 0.6, 0.6]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([1, 1, 1]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([1, 1, 1]), shadow=False)

        viewer = scene.create_viewer()  # type: ignore
        robot = teleconfig.ik_solver.robot.load(scene, "robot for ik")  # type: ignore
        while True:
            viewer.render()


@dataclass
@task_builder.register("vis_retargeting")
class vis_retargeting(TaskConfig):
    """
    visualize robot for retargeting
    """

    def run(self, path: Path):
        import sapien

        from vr_teleop.modules.teleconfig import TeleopConfig

        scene = sapien.Scene()

        teleconfig = TeleopConfig.from_yaml(path)

        scene = sapien.Scene()
        scene.set_ambient_light(np.array([0.6, 0.6, 0.6]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([1, 1, 1]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([1, 1, 1]), shadow=False)

        viewer = scene.create_viewer()  # type: ignore
        robot = teleconfig.retargeting.hand_model.load(scene, "robot for retargeting")  # type: ignore
        while True:
            viewer.render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("path", type=str)
    args, unknown = parser.parse_known_args()

    sys.argv = sys.argv[:1] + unknown

    cc = task_builder.default_configs[args.task]
    assert cc is not None
    assert issubclass(cc, TaskConfig)  # type: ignore
    cfg: TaskConfig = cc.from_cc()
    cfg.run(Path(args.path))


if __name__ == "__main__":
    main()
