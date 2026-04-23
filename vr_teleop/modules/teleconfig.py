from typing import TYPE_CHECKING

from ..configs.config import Config

from .ik import IKRobotConfig
from .retargeting import Retargeting, RetargetingConfig
from .tele import TeleConfig

if TYPE_CHECKING:
    from sapien import Scene
    from sapien.utils import Viewer
    from ..envs.base_env import Env
    from .tele import TeleOperatorBase
    from .vr import VRViewer

from dataclasses import dataclass

from sapien.utils import Viewer


@dataclass
class TeleopConfig(Config):
    viewer: str  # = "vr_gripper"
    tele: TeleConfig

    ik_solver: IKRobotConfig
    retargeting: RetargetingConfig | None = None

    ik_solver2: IKRobotConfig | None = None
    retargeting2: RetargetingConfig | None = None

    def create_teleop_modules(self, env: "Env", scene: "Scene", viewer: "VRViewer | Viewer") -> "TeleOperatorBase":
        order = self.tele.order
        tele_type = self.tele.config._type_

        robots = env._robots

        if "hand_body" in tele_type:
            if len(order) == 2:
                assert self.retargeting is not None and self.retargeting2 is not None
                retargetings = [
                    Retargeting(self.retargeting),
                    Retargeting(self.retargeting2),
                ]
            else:
                assert self.retargeting is not None
                retargetings = [Retargeting(self.retargeting)]
            ik_solvers = [self.ik_solver.create_ik_solver()]
        elif "hand" in tele_type:
            if len(order) == 2:
                assert self.retargeting is not None and self.retargeting2 is not None
                assert self.ik_solver2 is not None
                retargetings = [
                    Retargeting(self.retargeting),
                    Retargeting(self.retargeting2),
                ]
                ik_solvers = [
                    self.ik_solver.create_ik_solver(),
                    self.ik_solver2.create_ik_solver(),
                ]
            else:
                assert self.retargeting is not None
                retargetings = [Retargeting(self.retargeting)]
                ik_solvers = [self.ik_solver.create_ik_solver()]
        elif "gripper_body" in tele_type:
            ik_solvers = [self.ik_solver.create_ik_solver()]
            retargetings = None
        elif "gripper" in tele_type:
            if len(order) == 2:
                assert self.ik_solver2 is not None
                ik_solvers = [
                    self.ik_solver.create_ik_solver(),
                    self.ik_solver2.create_ik_solver(),
                ]
            else:
                ik_solvers = [self.ik_solver.create_ik_solver()]
            retargetings = None

        else:
            raise NotImplementedError

        teleop = self.tele.create_tele_operator(viewer, retargetings, ik_solvers, env)

        return teleop
