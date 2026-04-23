# type: ignore
from typing import TYPE_CHECKING

import numpy as np
import pinocchio
from sapien import Pose

if TYPE_CHECKING:
    from sapien.wrapper.pinocchio_model import PinocchioModel

    from .ik_solver import JointLimit


def compute_inverse_kinematics(
    self: "PinocchioModel",
    link_indexs: list[int],
    pose: tuple[Pose, ...],
    weights: np.ndarray | None = None,
    initial_qpos: np.ndarray | None = None,
    active_qmask: np.ndarray | None = None,
    eps: float = 1e-4,
    max_iterations: int = 1000,
    projector: "JointLimit | None" = None,
    dt: float = 0.1,
    damp: float = 1e-6,
):
    """
    The code is copied from sapien.wrapper.pinocchio_model.PinocchioModel.compute_inverse_kinematics
    """
    n_poses = len(pose)
    assert n_poses == len(link_indexs), "The number of poses should be equal to the number of link_indexs"

    for link_index in link_indexs:
        assert 0 <= link_index < len(self.link_id_to_frame_index)

    q = pinocchio.neutral(self.model) if initial_qpos is None else self.q_s2p(initial_qpos)  # type: ignore

    mask = weights if weights is not None else np.array(active_qmask)[self.index_p2s]

    mask = np.diag(mask)

    # compute joint for forward kinematics
    joints = []
    oMdes_list = []
    iMd_list = [None for _ in range(n_poses)]
    err = np.ones(n_poses * 6)  # shape=(n_links* 6)
    Jacobian = np.zeros((n_poses * 6, self.model.nv))  # shape=(n_links* 6, n_joints)

    for idx in range(n_poses):
        frame = int(self.link_id_to_frame_index[link_indexs[idx]])
        joint = self.model.frames[frame].parent
        joints.append(joint)

        T = pose[idx].to_transformation_matrix()  # target pose
        l2w = pinocchio.SE3()  # type: ignore
        l2w.translation[:] = T[:3, 3]
        l2w.rotation[:] = T[:3, :3]

        l2j = self.model.frames[frame].placement
        oMdes = l2w * l2j.inverse()
        oMdes_list.append(oMdes)

    best_error = 1e10
    best_q = np.array(q)

    for _i in range(max_iterations):
        pinocchio.forwardKinematics(self.model, self.data, q)  # type: ignore
        for idx in range(n_poses):
            iMd = self.data.oMi[joints[idx]].actInv(oMdes_list[idx])
            err[idx * 6 : idx * 6 + 6] = pinocchio.log6(iMd).vector  # type: ignore

            iMd_list[idx] = iMd

        err_norm = np.linalg.norm(err)

        if err_norm < best_error:
            best_error = err_norm
            best_q = q

        if err_norm < eps:
            success = True
            break

        # inverse kinematics
        for idx in range(n_poses):
            Jacob = pinocchio.computeJointJacobian(self.model, self.data, q, joints[idx])  # type: ignore
            Jlog = pinocchio.Jlog6(iMd_list[idx].inverse())  # type: ignore
            Jacob = -Jlog @ Jacob
            Jacobian[idx * 6 : idx * 6 + 6, ...] = Jacob

        Jacobian = Jacobian @ mask

        JJt = Jacobian @ Jacobian.T
        JJt[np.diag_indices_from(JJt)] += damp

        v = -(Jacobian.T @ np.linalg.solve(JJt, err))

        q = pinocchio.integrate(self.model, q, v * dt)  # type: ignore

        if projector is not None:
            q = projector.clip_pin_model(q)
    else:
        success = False

    return self.q_p2s(best_q), success, best_error
