from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sapien.physx import PhysxArticulation
    from sapien.wrapper.pinocchio_model import PinocchioModel


class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, model: "PinocchioModel", robot: "PhysxArticulation"):
        super().__init__()
        self.model = model
        self.joint_names = [j.name for j in robot.get_joints()]
        self.dof_joint_names = [j.name for j in robot.get_active_joints()]
        self.dof = robot.dof
        self.link_names = [l.name for l in robot.get_links()]
        self.joint_limits = np.array([j.limits[0] for j in robot.get_active_joints()])

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        # print(self.link_names)
        return self.link_names.index(name)

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: np.ndarray):
        self.model.compute_forward_kinematics(qpos)

    def get_link_pose(self, link_id: int) -> np.ndarray:
        return self.model.get_link_pose(link_id).to_transformation_matrix()

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> np.ndarray:
        return self.model.compute_single_link_local_jacobian(qpos, link_id)
