from dataclasses import dataclass, field
from ...configs.config import Config
from sapien import Pose


@dataclass
class PoseConfig(Config):
    p: tuple[float, float, float] = field(default_factory=lambda: (0, 0, 0))
    q: tuple[float, float, float, float] = field(default_factory=lambda: (1, 0, 0, 0))

    def to_sapien_pose(self):
        return Pose(self.p, self.q)
