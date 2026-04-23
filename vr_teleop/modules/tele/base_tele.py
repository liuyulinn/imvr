from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from ...configs.config import BuilderConfig, Config, ConfigRegistry


if TYPE_CHECKING:
    from sapien import Pose
    from sapien.utils import Viewer

    from ..ik import SeqIKSolver
    from ..retargeting import Retargeting
    from ..vr import VRViewerBase
    from .tele_flags import TeleFlags

_order2idx = {"left": 0, "right": 1}


@dataclass
class TeleOperatorConfig(Config): ...


class TeleOperatorBase:
    def __init__(
        self,
        config: TeleOperatorConfig,
        order: list[str],
        viewer: "Viewer | VRViewerBase",
        slices: dict[str, list[int]],
        action_limits: list[np.ndarray],
        delta_control: list[dict[str, bool]],
        normalize_action: list[dict[str, bool]],
        reset_state: bool,
        ik_solver: "list[SeqIKSolver]",
        retargeting: "list[Retargeting] | None",
    ) -> None:
        """
        Args:
            order: list of str("left", "right"), define the coresponsing order of robot to hand
            viewer: Viewer | VRViewer, viewer object
            slices: dict, slices of the action space
            action_limits: list of np.ndarray: [(D, 2)], action limits
            delta_control: list of dict, delta control
                Eg: [{"arm": 1, "hand": 1}],
                means the arm and hand both use delta control
            normalize_action: list of dict, normalize action
                Eg: [{"arm": 0, "hand": 0}],
                means the arm and hand both do not use normalize action
            ik_solver: list of SeqIKSolver, IK solver
            retargeting: list of Retargeting | None, retargeting
        """
        self.config = config
        self.order = order
        self.viewer = viewer
        self.slices = slices
        self.action_limits = action_limits

        self.delta_control = delta_control
        self.normalize_action = normalize_action

        self.reset_state = reset_state
        self.ik_solver = ik_solver
        self.retargeting = retargeting

        self.n_control = len(self.order)
        self.indices = [_order2idx[order] for order in self.order]

    def _normalize_action(self, action: np.ndarray, action_limits: np.ndarray) -> np.ndarray:
        """
        Normalize the action to [-1, 1]
        action: np.ndarry, (D, )
        action_limits: np.ndarray, (D, 2)
        """
        assert self.action_limits is not None, "Action limits should be provided"
        normalized_tensor = (action - action_limits[..., 0]) / (action_limits[..., 1] - action_limits[..., 0])
        normalized_tensor = np.clip(normalized_tensor, 0, 1)
        # print(action_limits)
        return normalized_tensor * 2 - 1

    def step(
        self,
        cur_qposs: list[np.ndarray],
        root_poses: "list[Pose]",
    ) -> "tuple[TeleFlags, list[np.ndarray]]":
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def action_components(self) -> dict[str, list[int]]:
        raise NotImplementedError


teleoperator_builder = ConfigRegistry()


@dataclass
class TeleOperatorBuilderConfig(BuilderConfig):
    def get_cfg(self) -> TeleOperatorConfig:
        config_cls, _ = teleoperator_builder._registry[self._type_]
        return config_cls.from_dict(self.config)
