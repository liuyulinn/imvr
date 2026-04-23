from typing import TYPE_CHECKING

import numpy as np
import sapien
from sapien import Pose, Scene
from sapien.utils import Viewer

if TYPE_CHECKING:
    from ..vr import VRHandViewer  # , VRViewer


class HandVisualizer:
    """
    A class to visualize rokoko hand tracker

    API:
        HandVisualizer.update_scene(joint_pos: np.array, wrist_pose: Pose = None)

        update scene
    """

    def __init__(
        self,
        scene: "Scene",
        viewer: "Viewer | VRHandViewer",  # | VRViewer",
        visualize_skeleton: bool = False,
        visualize_wrist_pose: bool = True,
    ) -> None:
        object.__init__(self)
        assert visualize_skeleton or visualize_wrist_pose, "must have something to visualize"
        self._if_visualize_skeleton = visualize_skeleton
        self._if_visualize_wrist_pose = visualize_wrist_pose

        self._scene = scene
        self._viewer = viewer
        self._renderer_context = sapien.render.SapienRenderer()._internal_context
        self._render_scene = self._viewer.render_scene
        self._create_visual_models()

    def _create_visual_models(self):
        sphere = self._renderer_context.create_uvsphere_mesh()
        cone = self._renderer_context.create_cone_mesh()
        capsule = self._renderer_context.create_capsule_mesh(0.1, 0.5, 16, 4)

        mat_red = self._renderer_context.create_material([5, 0, 0, 1], [0, 0, 0, 1], 0, 1, 0)
        mat_green = self._renderer_context.create_material([0, 1, 0, 1], [0, 0, 0, 1], 0, 1, 0)
        mat_blue = self._renderer_context.create_material([0, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0)
        mat_cyan = self._renderer_context.create_material([0, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0)
        mat_magenta = self._renderer_context.create_material([1, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0)

        self._red_sphere = self._renderer_context.create_model([sphere], [mat_red])
        self._green_sphere = self._renderer_context.create_model([sphere], [mat_green])
        self._blue_sphere = self._renderer_context.create_model([sphere], [mat_blue])
        self._cyan_sphere = self._renderer_context.create_model([sphere], [mat_cyan])
        self._magenta_sphere = self._renderer_context.create_model([sphere], [mat_magenta])

        self._finger_spheres = [
            self._red_sphere,
            self._green_sphere,
            self._blue_sphere,
            self._cyan_sphere,
        ]

        self._red_cone = self._renderer_context.create_model([cone], [mat_red])
        self._green_cone = self._renderer_context.create_model([cone], [mat_green])
        self._blue_cone = self._renderer_context.create_model([cone], [mat_blue])

        self._red_capsule = self._renderer_context.create_model([capsule], [mat_red])
        self._green_capsule = self._renderer_context.create_model([capsule], [mat_green])
        self._blue_capsule = self._renderer_context.create_model([capsule], [mat_blue])

        self._visual_models = None
        self._wrist_axes = None

    def _create_hand_models(self, joint_pos: np.ndarray):  # -> Node:
        """
        visualize hand skeleton
        """
        if self._render_scene is None:
            self._render_scene = self._viewer.render_scene
            assert self._render_scene is not None, "render_scene is None"
        render_scene = self._render_scene

        # wrist
        node = render_scene.add_node()
        obj = render_scene.add_object(self._magenta_sphere, node)
        obj.set_scale([0.01, 0.01, 0.01])
        obj.set_position(joint_pos[0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        # fingers
        for i in [1, 5, 9, 13, 17]:
            for j in range(4):
                obj = render_scene.add_object(self._finger_spheres[j], node)
                obj.set_scale([0.005, 0.005, 0.005])
                obj.set_position(joint_pos[i + j])
                obj.shading_mode = 0
                obj.cast_shadow = False

        lines = []
        for i in [1, 5, 9, 13, 17]:
            lines.extend(list(joint_pos[i + j]) + list(joint_pos[i + j + 1]) for j in range(3))
            lines.append(list(joint_pos[0]) + list(joint_pos[i]))

        line_set = self._renderer_context.create_line_set(lines, [1, 1, 1, 1, 1, 1, 1, 1] * 20)
        obj = render_scene.add_line_set(line_set, node)
        obj.set_scale([1, 1, 1])
        obj.line_width = 5

        return node

    def _update_visual_models(self, joint_pos: np.ndarray) -> None:
        if self._visual_models is not None:
            self._render_scene.remove_node(self._visual_models)

        self._visual_models = self._create_hand_models(joint_pos)

    def _creat_coordinate_axes(self):
        if self._render_scene is None:
            self._render_scene = self._viewer.render_scene
            assert self._render_scene is not None, "render_scene is None"

        render_scene = self._render_scene

        node = render_scene.add_node()
        obj = render_scene.add_object(self._red_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([1, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self._red_capsule, node)
        obj.set_position([0.52, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self._green_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 1, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self._green_capsule, node)
        obj.set_position([0, 0.51, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self._blue_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 0, 1])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self._blue_capsule, node)
        obj.set_position([0, 0, 0.5])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        node.set_scale([0.025, 0.025, 0.025])
        return node

    def _update_wrist_axes(self, wrist_pose: Pose):
        if self._wrist_axes is None:
            self._wrist_axes = self._creat_coordinate_axes()

        self._wrist_axes.set_position(wrist_pose.p)
        self._wrist_axes.set_rotation(wrist_pose.q)

    def update_scene(self, joint_pos: np.ndarray | None = None, wrist_pose: Pose | None = None):
        """
        Visualize the hand model
        """

        if self._if_visualize_skeleton:
            assert joint_pos is not None, "joint position must be specified"
            if isinstance(joint_pos, list):
                joint_pos = joint_pos[0]
            assert joint_pos is not None and joint_pos.shape == (21, 3), "joint position must be 21x3"
            self._update_visual_models(joint_pos)
        if self._if_visualize_wrist_pose:
            assert wrist_pose is not None, "wrist pose must be specified"
            self._update_wrist_axes(wrist_pose)
