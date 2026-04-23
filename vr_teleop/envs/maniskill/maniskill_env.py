import os

# from .base_env import ManiskillBaseEnv
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import cast

import gymnasium as gym
import numpy as np
import sapien
import torch
import yaml
from mani_skill.agents.controllers.passive_controller import PassiveControllerConfig
from mani_skill.agents.controllers.pd_base_vel import PDBaseForwardVelControllerConfig
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import DefaultMaterialsConfig, GPUMemoryConfig, SceneConfig, SimConfig
from mani_skill.utils.wrappers.record import RecordEpisode
from sapien.render import RenderBodyComponent

from ...configs.config import Config
from ..base_env import Env, EnvBaseConfig, Extra, env_builder
from ..utils.pose import PoseConfig


@dataclass
class NewSceneConfig(SceneConfig, Config):
    gravity: list[float] = field(default_factory=lambda: [0, 0, -9.81])  # type: ignore[brute force override]


@dataclass
class NewSimConfig(SimConfig, Config):
    spacing: int = 20
    """Controls the spacing between parallel environments when simulating on GPU in meters. Increase this value
    if you expect objects in one parallel environment to impact objects within this spacing distance"""

    sim_freq: int = 100
    """simulation frequency (Hz)"""

    control_freq: int = 20  # | None = None  # type: ignore
    """
    control frequency (Hz). Every control step (e.g. env.step) contains sim_freq / control_freq physx simulation steps
    NOTE: you can either set control_freq in sim_config or the control_freq in agent's controller config. If you have set
    both, the value should be exactly the same.
    """

    gpu_memory_config: GPUMemoryConfig = field(default_factory=GPUMemoryConfig)
    scene_config: NewSceneConfig = field(default_factory=NewSceneConfig)  # pyright: ignore[reportIncompatibleVariableOverride]
    default_materials_config: DefaultMaterialsConfig = field(default_factory=DefaultMaterialsConfig)


@dataclass
class ManiskillEnvConfig(EnvBaseConfig):
    env_name: str = "PickCube-v1"
    robot_uids: str | None = None

    ignore_terminations: bool = True  # ignore the terminations for all mani_skill envs

    obs_mode: str = "state"
    action_mode: str | None = None
    render_mode: str = "rgb_array"  # use rgb_array for teleoperation, "human" for human rendering
    sim_backend: str = "cpu"

    control_mode: str = "pd_pos_control"
    sim_config: NewSimConfig = field(default_factory=NewSimConfig)

    root_pose: PoseConfig = PoseConfig()
    goal_site: PoseConfig = PoseConfig(p=[0.0, 0.0, 1.0])

    success_condition: bool = True  # "whether have success condition or not"
    auto_record_success: bool = False
    auto_record_success_threshold: int = 20


@env_builder.register(ManiskillEnvConfig, "maniskill_env")
class ManiskillEnv(Env):
    def __init__(self, num_envs: int, config: ManiskillEnvConfig):
        super().__init__(num_envs, config)
        self.config = config
        env_name = config.env_name
        ignore_terminations = config.ignore_terminations
        max_episode_steps = 5000  # config.max_episode_steps

        sim_config = deepcopy(config.sim_config)
        sim_config.scene_config.gravity = np.array(sim_config.scene_config.gravity)
        sim_config = asdict(sim_config)

        extra_args = {"robot_uids": config.robot_uids} if config.robot_uids is not None else {}
        if config.action_mode is not None:
            extra_args["control_mode"] = config.action_mode

        self._env = cast(
            BaseEnv,
            gym.make(
                env_name,
                num_envs=num_envs,
                sim_config=sim_config,
                reconfiguration_freq=0,
                render_mode=config.render_mode,
                sim_backend=config.sim_backend,
                control_mode=config.control_mode,
                **extra_args,
            ),
        )
        assert isinstance(self._env.unwrapped, BaseEnv)
        self.ignore_terminations = ignore_terminations
        self.num_envs = int(self._env.num_envs)

        self.returns = torch.zeros(self.num_envs, device=self.base_env.device)
        self.max_episode_steps = max_episode_steps
        print("max_episode_steps", self.max_episode_steps)

        # add record
        self.output_dir = f"records/maniskill/{env_name}/{time.strftime('%Y%m%d-%H%M%S')}"
        self._env = RecordEpisode(
            self._env,
            output_dir=self.output_dir,
            trajectory_name="trajectory",
            save_video=False,
            info_on_video=False,
            source_type="teleoperation",
            source_desc="VR teleoperation",
        )

        if getattr(self._env, "goal_site", None) is None:
            goal_site_pose = config.goal_site.to_sapien_pose()

            builder = self.get_cpu_scene().create_actor_builder()

            builder.add_sphere_visual(radius=0.02, material=[0, 1, 0])
            goal_site = builder.build_static(name="goal_site")
            goal_site.set_pose(goal_site_pose)
        else:
            goal_site = self._env.goal_site._objs[0]

        self.goal_site_render_body = goal_site.find_component_by_type(RenderBodyComponent)  # type: ignore

        self.use_success_condition = config.success_condition
        self.auto_record_success = config.auto_record_success

        if self.auto_record_success:
            self.success_condition = 0
            self.auto_record_success_threshold = config.auto_record_success_threshold

    def get_cpu_scene(self) -> sapien.Scene:
        assert len(self.base_env.scene.sub_scenes) == 1
        return self.base_env.scene.sub_scenes[0]

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> BaseEnv:
        return cast(BaseEnv, self._env.unwrapped)

    def reset(self, save: bool = True, seed: int | list[int] | None = None, options: dict | None = None):
        options = options or {}

        obs, info = self._env.reset(seed=seed, options=options, save=save)

        # reset to origin color
        self.goal_site_render_body.render_shapes[0].parts[0].get_material().base_color = [0, 1, 0, 1]
        if "env_idx" in options:
            env_idx = options["env_idx"]
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.base_env.device)
            mask[env_idx] = True
            self.returns[mask] = 0
        else:
            self.returns *= 0
        return obs, info

    def step(
        self, action: list[np.ndarray]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Extra]:
        action = torch.from_numpy(np.concatenate(action, axis=0))[None, ...]  # type: ignore

        next_obs, reward, terminated, truncated, info = self._env.step(action)

        if self.use_success_condition and info["success"]:
            self.goal_site_render_body.render_shapes[0].parts[0].get_material().base_color = [0, 0, 1, 1]  # type: ignore
        else:
            self.goal_site_render_body.render_shapes[0].parts[0].get_material().base_color = [1, 0, 0, 1]

        terminated = False

        if self.use_success_condition and self.auto_record_success:
            if info["success"] and self.success_condition == 0:
                self.success_condition = 1

            elif not info["success"]:
                self.success_condition = 0

            elif self.success_condition:
                self.success_condition += 1

            if self.success_condition > self.auto_record_success_threshold:
                terminated = True
                self.success_condition = 0

        return next_obs, reward, terminated, truncated, info

    def __del__(self):
        # if no file store, delete the folder
        if not os.path.exists(os.path.join(self.output_dir, "trajectory.json")):
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        self._env.close()

        return

    def render(self, env_ids: list[int] | None = None, spacing: float = 5.0):
        if self.base_env._viewer is None:
            from sapien.utils import Viewer

            self.base_env._viewer = Viewer()
        if len(self.base_env._viewer.scenes) == 0:
            self._setup_viewer(env_ids, spacing)

        self.base_env.render_human()

    def _setup_viewer(self, idx: list[int] | np.ndarray | None = None, spacing: float = 5.0):
        self.base_env._setup_viewer()

    @property
    def _robots(self):
        """Warning: in this case, we only support one robot"""
        return [self.base_env.agent]

    @property
    def agent_qpos(self):
        return [self.base_env.agent.robot.qpos[0].numpy()]

    @property
    def agent_root_pose(self):
        robot = self.base_env.agent.robot
        return [sapien.Pose(p=robot.root_pose.p[0].tolist(), q=robot.root_pose.q[0].tolist())]

    # @property
    # def agent_control_qlimits(self):
    #     return [self.base_env.agent.robot.qlimits[0].numpy()]

    @property
    def agent_active_joint_names(self) -> list[list[str]]:
        return [[i.name for i in self.base_env.agent.robot.get_active_joints()]]

    @property
    def agent_controlled_joint_names(self) -> list[dict[str, list[str]]]:
        controller_joint_names = []
        for robot in self._robots:
            controller_joint_names_dict = {}
            active_joints = robot.robot.get_active_joints()
            for key, value in robot.controller.controllers.items():
                controller_joint_names_dict[key] = [active_joints[i].name for i in value.active_joint_indices]
            controller_joint_names.append(controller_joint_names_dict)
        return controller_joint_names

    @property
    def agent_control_status(self) -> tuple[list[dict[str, bool]], list[dict[str, bool]], list[np.ndarray]]:
        normalize_status_list = []
        use_delta_list = []
        joint_limits_list = []
        for robot in self._robots:
            normalize_status = {}
            use_delta = {}
            joint_limit = {}

            # cases where arm & gripper are using separate controller
            for key, value in robot.controller.controllers.items():
                if isinstance(value.config, PassiveControllerConfig):
                    continue
                elif isinstance(value.config, PDBaseForwardVelControllerConfig):
                    controller_config = value.config
                    normalize_status[key] = controller_config.normalize_action
                    # joint_limit.append(value._get_joint_limits())
                    continue
                # import pdb; pdb.set_trace()
                controller_config = value.config
                normalize_status[key] = controller_config.normalize_action
                use_delta[key] = controller_config.use_delta
                limits = value._get_joint_limits()
                limits[np.isneginf(limits)] = -6.28
                limits[np.isposinf(limits)] = 6.28
                joint_limit[key] = limits
            normalize_status_list.append(normalize_status)
            use_delta_list.append(use_delta)
            joint_limits_list.append(joint_limit)  # np.concatenate(joint_limit))
        return normalize_status_list, use_delta_list, joint_limits_list

    @property
    def goal_site(self):
        return self._env.goal_site  # type: ignore

    @property
    def object(self):
        return self._env.obj

    @property
    def control_freq(self):
        return self.config.sim_config.control_freq

    @property
    def viewer(self):
        if (viewer := self.base_env.viewer) is None:
            from sapien.utils import Viewer

            viewer = self.base_env._viewer = Viewer()
        return viewer

    def set_viewer(self, viewer):
        self.base_env._viewer = viewer

    def step_action(self, action: np.ndarray, idx: int):
        action = torch.from_numpy(action)[None, ...]  # type: ignore
        self._robots[idx].set_action(action)

    def set_qpos(self, qpos: np.ndarray, idx: int):
        qpos = torch.from_numpy(qpos)[None, ...]  # type: ignore
        self._robots[idx].robot.set_qpos(qpos)

    def sample_random_action(self):
        """
        Sample a random action for each robot in the environment.
        Returns:
            A list of random actions, one for each robot.
        """
        return self._env.action_space.sample()
