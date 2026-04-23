from dataclasses import dataclass

import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from torch import Tensor

from ..configs.config import BuilderConfig, Config, ConfigRegistry


@dataclass
class Extra:
    """
    Use the convention in mani_skill
    final_info stores the original return of each env's step
        (including the ending of the episode)
    while info may store info (including those reset)

    final_obs store the observation of the terminated envs only
    """

    info: dict
    final_obs: dict[str, torch.Tensor] | None = None
    final_info: dict | None = None


@dataclass
class EnvBaseConfig(Config): ...


class Env:
    base_env: BaseEnv

    def __init__(self, num_envs, config: EnvBaseConfig) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.config = config

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    def get_obs(self, mode: str | None = None):
        raise NotImplementedError

    def get_state(self, keys: dict | None = None, pretty: bool = False):
        raise NotImplementedError

    def set_state(self, state: dict):
        raise NotImplementedError

    def step(self, action: list[np.ndarray]) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Extra]:
        raise NotImplementedError

    def render_image(self):
        raise NotImplementedError

    def render(self, env_ids: list[int] | None = None):
        raise NotImplementedError

    def sample_random_action(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def get_cpu_scene(self):
        raise NotImplementedError

    @property
    def viewer(self):
        raise NotImplementedError

    @property
    def _robots(self):
        """
        return a list of robots
        """

        raise NotImplementedError

    @property
    def agent_qpos(self):
        """
        return list of np.ndarray (D,)
        """
        raise NotImplementedError

    @property
    def agent_root_pose(self):
        """
        return list of sapien.Pose
        """
        raise NotImplementedError

    @property
    def agent_qlimits(self):
        """
        return list of np.ndarray (D, 2)
        """
        raise NotImplementedError

    @property
    def agent_active_joint_names(self) -> list[list[str]]:
        """
        return list of list of str
        """
        raise NotImplementedError

    @property
    def agent_controlled_joint_names(self) -> list[dict[str, list[str]]]:
        """
        return list of dict
        """
        raise NotImplementedError

    def get_controlled_joint_names(self, mode: str, idx: int) -> list[str]:
        """
        mode: str
        idx: int
        """

        return self.agent_controlled_joint_names[idx].get(mode, [])

    @property
    def agent_control_status(self) -> tuple[list[dict[str, bool]], list[dict[str, bool]], list[np.ndarray]]:
        """
        return normalize_action, use_delta
        Eg: [{"arm": 1, "hand": 1}], [{"arm": 0, "hand": 0}],
            means the arm and hand both use normalize action,
            and do not use delta control
        """
        raise NotImplementedError

    def step_action(self, action: np.ndarray, idx: int):
        """
        action: np.ndarray (D,)
        idx: int
        """
        raise NotImplementedError

    def clean_warning_error(self, idx: int):
        pass

    @property
    def goal_site(self):
        """
        return an actor"""
        raise NotImplementedError

    @property
    def control_freq(self):
        raise NotImplementedError

    def set_viewer(self, viewer, scene=None):
        raise NotImplementedError

    @classmethod
    def load(cls, state):
        env = cls.__new__(cls)
        env.__setstate__(state)
        return env

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def dump(self):
        return self.__getstate__()


env_builder = ConfigRegistry()


@dataclass
class EnvBuilderConfig(BuilderConfig):
    # _type_: str = "maniskill_env"
    """
    Support the following types:
    - maniskill_env
    - real_env
    """

    def get_cfg(self) -> EnvBaseConfig:
        config_cls, _ = env_builder._registry[self._type_]
        return config_cls.from_dict(self.config)
