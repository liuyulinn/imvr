import json
from pathlib import Path

import numpy as np
import sapien
from sapien import Pose, Scene
from sapien.render import (
    RenderVRDisplay,
)

# from .VRBase import VRViewerBase
from vr_teleop.modules.vr.vr_base import VRViewerBase

sapien.render.enable_vr()
# sapien.render.set_log_level("info")
sapien.render.set_viewer_shader_dir("../vulkan_shader/vr_default")
controller_id2names = {1: "left", 2: "right"}

_hand_pos_index = range(0, 25)
_hand_pos_index = [i for i in _hand_pos_index if i not in [5, 10, 15, 20]]


class VRHandViewer(VRViewerBase):
    def __init__(self, visualize: bool = True):
        super().__init__(visualize=visualize)

        action_file = Path.home() / ".sapien" / "steamvr_actions.json"
        action_file.parent.mkdir(parents=True, exist_ok=True)
        # if not action_file.exists():
        with action_file.open("w") as f:
            json.dump(
                {
                    "version": 1,
                    "minimum_required_version": 1,
                    "default_bindings": [
                        {
                            "controller_type": "oculus_touch",
                            "binding_url": "oculus_touch.json",
                        }
                    ],
                    "actions": [
                        {
                            "name": "/actions/global/in/HandSkeletonLeft",
                            "type": "skeleton",
                            "skeleton": "/skeleton/hand/left",
                        },
                        {
                            "name": "/actions/global/in/HandSkeletonRight",
                            "type": "skeleton",
                            "skeleton": "/skeleton/hand/right",
                        },
                    ],
                    "action_sets": [{"name": "/actions/global", "usage": "leftright"}],
                    "localization": [
                        {
                            "language_tag": "en_US",
                            "/actions/global": "Global",
                            "/actions/global/in/HandPoseLeft": "Hand Pose Left",
                            "/actions/global/in/HandPoseRight": "Hand Pose Right",
                            "/actions/global/in/HandSkeletonLeft": "Hand Skeleton Left",
                            "/actions/global/in/HandSkeletonRight": "Hand Skeleton Right",
                        }
                    ],
                },
                f,
            )
        with (Path.home() / ".sapien" / "oculus_touch.json").open("w") as f:
            json.dump(
                {
                    "action_manifest_version": 1,
                    "bindings": {
                        "/actions/global": {
                            "skeleton": [
                                {
                                    "output": "/actions/global/in/handskeletonleft",
                                    "path": "/user/hand/left/input/skeleton/left",
                                },
                                {
                                    "output": "/actions/global/in/handskeletonright",
                                    "path": "/user/hand/right/input/skeleton/right",
                                },
                            ],
                            "sources": [],
                        }
                    },
                    "controller_type": "oculus_touch",
                    "description": "Default Oculus Touch bindings for SteamVR Home.",
                    "name": "Default Oculus Touch Bindings",
                },
                f,
            )
        sapien.render.set_vr_action_manifest_filename(str(action_file))
        print(sapien.render.get_vr_action_manifest_filename())

        self.visualize = visualize
        self.vr = RenderVRDisplay()
        self.controllers = self.vr.get_controller_ids()
        self.renderer_context = sapien.render.SapienRenderer()._internal_context
        self._create_visual_models()

        self.reset()

    def reset(self):
        self._controller_axes = None
        self.marker_spheres = None

        self.left_hand_spheres = None
        self.right_hand_spheres = None

    def set_scene(self, scene: Scene):
        self.scene = scene
        self.vr.set_scene(scene)

    @property
    def scenes(self):
        return [self.scene]

    @property
    def root_pose(self):
        return self.vr.root_pose

    @root_pose.setter
    def root_pose(self, pose):
        self.vr.root_pose = pose

    @property
    def head_pose(self):
        """
        return the head pose of the VR viewer
        """
        return self.vr.get_hmd_pose()

    @property
    def controllers_names(self):  # , controller_id):
        """
        return the controllers names
            1: left controller
            2: right controller
        """
        return controller_id2names  # [controller_id]

    @property
    def controller_poses(self):
        """
        return the controller poses, [left_pose, right_pose]
        with respect to the headset frame
        pose: in format of sapien.Pose, ([x, y, z], [qx, qy, qz, qw]), the position and quaternion of the root pose
        """
        return [self.vr.get_controller_pose(c) for c in self.controllers]

    @property
    def controller_left_poses(self):
        """
        return the left controller pose
        """
        return self.vr.get_controller_pose(self.controllers[0])

    @property
    def controller_right_poses(self):
        """
        return the right controller pose
        """
        return self.vr.get_controller_pose(self.controllers[1])

    @property
    def controller_axes(self):
        return [self.vr.get_controller_axis_state(c, 0) for c in self.controllers]

    @property
    def controller_left_axes(self):
        return self.vr.get_controller_axis_state(self.controllers[0], 0)

    @property
    def controller_right_axes(self):
        return self.vr.get_controller_axis_state(self.controllers[1], 0)

    """'wrapper functions for the controller poses"""

    @property
    def controller_base_poses(self):
        """
        controller base pose with respect to the world frame
        """
        t2ws = [self.root_pose * c2r for c2r in self.controller_poses]
        return t2ws

    @property
    def controller_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]  # type: ignore
        t2ws = [self.root_pose * c2r * t2c for c2r in self.controller_poses]

        # c2r = self.controller_poses
        # r2w = self.root_pose
        # t2w = r2w * c2r * t2c
        # print(t2w)
        return t2ws

    @property
    def controller_left_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]  # type: ignore
        c2r = self.controller_poses[0]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        return t2w

    @property
    def controller_right_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]  # type: ignore
        c2r = self.controller_poses[1]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        return t2w

    @property
    def left_hand_root_pose(self) -> Pose:
        """
        return the root pose of the left hand
        """
        # print(self.vr.get_left_hand_skeletal_poses())
        skeleton_pose = self.vr.get_left_hand_skeletal_poses()
        if len(skeleton_pose) != 31:
            return None  # pyright: ignore
        return (
            self.root_pose
            * self.vr.get_left_hand_root_pose()
            * self.vr.get_left_hand_skeletal_poses()[1]
            * Pose(q=[0, 0, 0.7071068, 0.7071068])
            * Pose(q=[0, 1, 0, 0])
        )

    @property
    def left_hand_pose(self) -> np.ndarray | None:
        """
        return the left hand pose, (21, 3)
        """
        hand_skeletal_poses = self.vr.get_left_hand_skeletal_poses()[1:26]  # exclude the root pose

        hand_pos = []
        for i, p in enumerate(hand_skeletal_poses):
            hand_skeletal_poses[i] = self.root_pose * self.vr.get_left_hand_root_pose() * p
            hand_pos.append(hand_skeletal_poses[i].p)

        if len(hand_pos) != 25:
            return None
        hand_pos = np.array(hand_pos)[_hand_pos_index]
        return hand_pos

    @property
    def right_hand_root_pose(self) -> Pose | None:
        """
        return the root pose of the right hand
        """
        skeleton_pose = self.vr.get_right_hand_skeletal_poses()
        if len(skeleton_pose) != 31:
            return None
        return (
            self.root_pose
            * self.vr.get_right_hand_root_pose()
            * self.vr.get_right_hand_skeletal_poses()[1]
            * Pose(q=[0, 0, 0.7071068, 0.7071068])
            # * Pose(
            #     q=[
            #         0.7071068,
            #         -0.7071068,
            #         0,
            #         0,
            #     ]
            # )
            # * Pose(q=[0, 0, 0, 1])
        )

    @property
    def right_hand_pose(self) -> np.ndarray | None:
        """
        return the right hand pose, (21, 3)
        """
        hand_skeletal_poses = self.vr.get_right_hand_skeletal_poses()[1:26]
        hand_pos = []
        for i, p in enumerate(hand_skeletal_poses):
            hand_skeletal_poses[i] = self.root_pose * self.vr.get_right_hand_root_pose() * p
            hand_pos.append(hand_skeletal_poses[i].p)

        if len(hand_pos) != 25:
            return None
        hand_pos = np.array(hand_pos)[_hand_pos_index]
        return hand_pos

    @property
    def hand_root_pose(self) -> list[Pose | None]:
        """
        return the left and right hand root pose
        """
        return [self.left_hand_root_pose, self.right_hand_root_pose]

    @property
    def hand_pose(self) -> list[np.ndarray | None]:
        """
        return the left and right hand pose, (31, 7)
        """
        return [self.left_hand_pose, self.right_hand_pose]

    @property
    def render_scene(self):
        return self.vr._internal_scene

    def pressed_button(self, controller_id: int) -> str:
        """
        return the button pressed status
        controller_id: the id of the controller, [1, 2]
        return: None, not pressed;
                'up', upper button pressed;
                'down', lower button pressed;
        """
        raise NotImplementedError

    def render(self):
        if self.visualize:
            self._update_controller_axes()
            self.update_hand_skeleton()
        self.vr.update_render()
        self.vr.render()

    @property
    def ray_angle(self):
        return np.pi / 4

    def pick(self, index):
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]  # type: ignore
        c2r = self.controller_poses[index]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        d = t2w.to_transformation_matrix()[:3, 0]

        assert isinstance(self.scene.physx_system, sapien.physx.PhysxCpuSystem)
        px: sapien.physx.PhysxCpuSystem = self.scene.physx_system
        res = px.raycast(t2w.p, d, 50)
        return res

    def update_marker_sphere(self, i, hit):
        if self.marker_spheres is None:
            self.marker_spheres = [self._create_marker_sphere() for c in self.controllers]

        if hit is None:
            self.marker_spheres[i].transparency = 1
        else:
            self.marker_spheres[i].set_position(hit.position)
            self.marker_spheres[i].transparency = 0

    def update_hand_skeleton(self):
        root_pose = self.root_pose
        hrp = self.vr.get_left_hand_root_pose()
        poses = self.vr.get_left_hand_skeletal_poses()
        if len(poses) != 31:
            return

        if self.left_hand_spheres is None:
            self.left_hand_spheres = [self._create_hand_sphere() for _ in range(25)]

        for s, p in zip(self.left_hand_spheres, poses[1:]):
            sphere_pose = root_pose * hrp * p
            s.transparency = 0
            s.set_position(sphere_pose.p)

        hrp = self.vr.get_right_hand_root_pose()
        poses = self.vr.get_right_hand_skeletal_poses()
        if len(poses) != 31:
            return

        if self.right_hand_spheres is None:
            self.right_hand_spheres = [self._create_hand_sphere() for _ in range(25)]

        for s, p in zip(self.right_hand_spheres, poses[1:]):
            sphere_pose = root_pose * hrp * p
            s.transparency = 0
            s.set_position(sphere_pose.p)

    # helper visuals
    def _create_visual_models(self):
        self.cone = self.renderer_context.create_cone_mesh(16)
        self.capsule = self.renderer_context.create_capsule_mesh(0.1, 0.5, 16, 4)
        self.cylinder = self.renderer_context.create_cylinder_mesh(16)
        self.sphere = self.renderer_context.create_uvsphere_mesh()

        self.laser = self.renderer_context.create_line_set([0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0])

        self.mat_red = self.renderer_context.create_material([5, 0, 0, 1], [0, 0, 0, 1], 0, 1, 0)
        self.mat_green = self.renderer_context.create_material([0, 1, 0, 1], [0, 0, 0, 1], 0, 1, 0)
        self.mat_blue = self.renderer_context.create_material([0, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0)
        self.mat_cyan = self.renderer_context.create_material([0, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0)
        self.mat_magenta = self.renderer_context.create_material([1, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0)
        self.mat_white = self.renderer_context.create_material([1, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0)
        self.red_cone = self.renderer_context.create_model([self.cone], [self.mat_red])
        self.green_cone = self.renderer_context.create_model([self.cone], [self.mat_green])
        self.blue_cone = self.renderer_context.create_model([self.cone], [self.mat_blue])
        self.red_capsule = self.renderer_context.create_model([self.capsule], [self.mat_red])
        self.green_capsule = self.renderer_context.create_model([self.capsule], [self.mat_green])
        self.blue_capsule = self.renderer_context.create_model([self.capsule], [self.mat_blue])
        self.cyan_capsule = self.renderer_context.create_model([self.capsule], [self.mat_cyan])
        self.magenta_capsule = self.renderer_context.create_model([self.capsule], [self.mat_magenta])
        self.white_cylinder = self.renderer_context.create_model([self.cylinder], [self.mat_white])
        self.red_sphere = self.renderer_context.create_model([self.sphere], [self.mat_red])
        self.cyan_sphere = self.renderer_context.create_model([self.sphere], [self.mat_cyan])

    def _create_coordiate_axes(self):
        render_scene = self.render_scene

        node = render_scene.add_node()
        obj = render_scene.add_object(self.red_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([1, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.red_capsule, node)
        obj.set_position([0.52, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 1, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_capsule, node)
        obj.set_position([0, 0.51, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 0, 1])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_capsule, node)
        obj.set_position([0, 0, 0.5])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        # obj = render_scene.add_line_set(self.laser, node)
        # obj.set_scale([40, 0, 0])
        # obj.line_width = 20
        # ray_pose = sapien.Pose()
        # ray_pose.rpy = [0, self.ray_angle, 0]
        # obj.set_rotation(ray_pose.q)

        node.set_scale([0.025, 0.025, 0.025])

        return node

    def _update_controller_axes(self):
        if self._controller_axes is None:
            self._controller_axes = [self._create_coordiate_axes() for c in self.controllers]

        for n, pose in zip(self._controller_axes, self.hand_root_pose):
            # print(pose)
            if pose is not None:
                # c2w = self.vr.root_pose * pose
                n.set_position(pose.p)
                n.set_rotation(pose.q)

    def _create_marker_sphere(self):
        node = self.render_scene.add_object(self.red_sphere)
        node.set_scale([0.05] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node

    def _create_hand_sphere(self):
        node = self.render_scene.add_object(self.cyan_sphere)
        node.set_scale([0.005] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node


if __name__ == "__main__":
    import torch

    scene = sapien.Scene()

    viewer = VRHandViewer(visualize=True)
    viewer.set_scene(scene)

    Stop = False
    while not Stop:
        viewer.render()

        if viewer.left_hand_pose is not None:
            print("streaming....")
            print(viewer.left_hand_pose)
