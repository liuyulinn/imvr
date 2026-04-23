import os
from pathlib import Path

import numpy as np
import sapien
from sapien import Pose, Scene
from sapien.render import (
    RenderVRDisplay,
)

sapien.render.set_viewer_shader_dir("../vulkan_shader/vr_default")
controller_id2names = {1: "left", 2: "right"}

sapien.render.enable_vr()


class VRViewerBase:
    def __init__(self, visualize: bool = True):
        self.visualize = visualize
        self.vr = RenderVRDisplay()
        self.controllers = self.vr.get_controller_ids()
        self.renderer_context = sapien.render.SapienRenderer()._internal_context

        if os.path.exists(Path.home() / ".sapien" / "steamvr_actions.json"):
            os.remove(Path.home() / ".sapien" / "steamvr_actions.json")
        if os.path.exists(Path.home() / ".sapien" / "oculus_touch.json"):
            os.remove(Path.home() / ".sapien" / "oculus_touch.json")

        self._create_visual_models()

        self.reset()

    def reset(self):
        self._controller_axes = None
        self.marker_spheres = None

    def set_scene(self, scene: Scene):
        """
        register the VR viewer to the scene
        """
        self.scene = scene
        self.vr.set_scene(scene)

    def align_camera_to_robot(self):
        # fixme: set root pose to robot pose
        """
        align the camera to the robot root, used for PicoVR
        """
        ...

    def update_ik_range_aabb(self):
        """
        update the ik range aabb
        """
        ...

    def get_contacts(self):
        return self.scene.get_contacts()

    def close(self):
        del self.controllers
        del self.vr
        del self.renderer_context

    @property
    def scenes(self):
        return [self.scene]

    @property
    def root_pose(self):
        """
        return the root pose of the VR viewer
        see set_root_pose for more details
        """
        return self.vr.root_pose

    @root_pose.setter
    def root_pose(self, pose: Pose):
        """
        Set the root pose of the VR viewer
        pose: sapien.Pose, ([x, y, z], [qx, qy, qz, qw]), the position and quaternion of the root pose
        root_pose: the root pose of the VR viewer. which is the foot of the VR viewer
        """
        self.vr.root_pose = pose

    @property
    def head_pose(self):
        """
        return the head pose of the VR viewer
        """
        return self.vr.get_hmd_pose()

    @property
    def controllers_names(self):
        """
        return the controllers names
            1: left controller
            2: right controller
        """
        return controller_id2names

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
        t2c.p = [-0.1, 0, 0]
        t2c.rpy = [0, self.ray_angle, 0]  # type: ignore
        t2ws = [self.root_pose * c2r * t2c for c2r in self.controller_poses]

        return t2ws

    @property
    def controller_left_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.p = [-0.1, 0, 0]
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
        t2c.p = [-0.1, 0, 0]
        t2c.rpy = [0, self.ray_angle, 0]  # type: ignore
        c2r = self.controller_poses[1]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        return t2w

    @property
    def hand_root_pose(self) -> list[Pose]:
        """
        return the hand root pose of the controller
        controller_id: the id of the controller, [1, 2]
        """
        raise NotImplementedError

    @property
    def hand_pose(self) -> list[np.ndarray | None]:
        raise NotImplementedError

    @property
    def render_scene(self):
        """
        return the left and right hand pose, [(21, 3)]
        """
        return self.vr._internal_scene

    def up_button_state(self, controller_id: int) -> float:
        """
        return the up button states
        controller_id: the id of the controller, [1, 2]
        return: 0, not pressed;
                1, pressed;
        """
        return self.vr.get_controller_axis_state(controller_id, 1)[0]  # up button

    def down_button_state(self, controller_id: int) -> float:
        return self.vr.get_controller_axis_state(controller_id, 2)[0]  # down button

    def pressed_button(self, controller_id: int) -> str:
        """
        return the button pressed status
        controller_id: the id of the controller, [1, 2]
        return: None, not pressed;
                'up', upper button pressed;
                'down', lower button pressed;
        """
        button_pressed = self.vr.get_controller_button_pressed(controller_id)
        button_mapping = {8589934592: "up", 17179869188: "down", 25769803780: "both", 2: "B", 128: "A"}
        return button_mapping.get(button_pressed, "null")

    def render(self):
        """
        update the VR viewer
        """
        if self.visualize:
            self._update_controller_axes()
        # self.scene.update_render()
        self.vr.update_render()
        self.vr.render()

    @property
    def ray_angle(self):
        return np.pi / 4

    def update_marker_sphere(self, i, hit):
        """
        update the marker sphere which is used to show the target of ray casted from the controller
        """
        if self.marker_spheres is None:
            self.marker_spheres = [self._create_marker_sphere() for c in self.controllers]

        if hit is None:
            self.marker_spheres[i].transparency = 1  # total transparent, invisible
        else:
            self.marker_spheres[i].set_position(hit.position)
            self.marker_spheres[i].transparency = 0

    """help visual functions"""

    def _create_visual_models(self):
        self._create_coordinate_models()

    def _create_coordinate_models(self):
        """
        Create coordinate axes models
        """
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

        node.set_scale([0.025, 0.025, 0.025])

        return node

    def _update_controller_axes(self):
        if self._controller_axes is None:
            self._controller_axes = [self._create_coordiate_axes() for c in self.controllers]

        for n, pose in zip(self._controller_axes, self.controller_hand_poses):
            c2w = pose
            n.set_position(c2w.p)
            n.set_rotation(c2w.q)

    def _create_marker_sphere(self):
        node = self.render_scene.add_object(self.red_sphere)
        node.set_scale([0.05] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node


if __name__ == "__main__":
    scene = sapien.Scene()

    viewer = VRViewerBase()
    viewer.set_scene(scene)

    while True:
        print(viewer.pressed_button(1))
        print(viewer.pressed_button(2))
        print(viewer.vr.get_controller_button_touched(1))
        print(viewer.controller_axes)
        controller_id = 2

        print("0", viewer.vr.get_controller_axis_state(controller_id, 0))
        print("1", viewer.vr.get_controller_axis_state(controller_id, 1))
        print("2", viewer.vr.get_controller_axis_state(controller_id, 2))
        print("3", viewer.vr.get_controller_axis_state(controller_id, 3))
        print("4", viewer.vr.get_controller_axis_state(controller_id, 4))
        viewer.render()
