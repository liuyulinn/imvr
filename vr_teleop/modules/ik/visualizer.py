from typing import TYPE_CHECKING

import numpy as np
import sapien
from sapien import Pose, Scene
from sapien.utils import Viewer

if TYPE_CHECKING:
    from ..vr import VRViewerBase


class RotationVisualizer:
    """
    A class to visualize multiple rotations

    API:
        RotationVisualizer.update_scene(rotations: np.array)

        update scene
    """

    def __init__(
        self,
        scene: Scene,
        viewer: "Viewer | VRViewerBase",
    ) -> None:
        object.__init__(self)

        self._scene = scene
        self._viewer = viewer
        self._renderer_context = sapien.render.SapienRenderer()._internal_context
        self._render_scene = self._viewer.render_scene
        self._create_visual_models()

    def _create_visual_models(self):
        # self.cylinder = self.renderer_context.create_cylinder_mesh(16)
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
        # self.white_sphere = self._renderer_context.create_model(
        #     [self.sphere], [self.mat_white]
        # )
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

        self._wrist_axes = []

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

    def _update_axes(self, pose: Pose, idx: int):
        if len(self._wrist_axes) <= idx:
            self._wrist_axes.append(self._creat_coordinate_axes())

        self._wrist_axes[idx].set_position(pose.p)
        self._wrist_axes[idx].set_rotation(pose.q)

    def update_scene(self, poses: list[Pose]):
        """
        Visualize the hand model
        """
        for idx in range(len(poses)):
            self._update_axes(poses[idx], idx)
