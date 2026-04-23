from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List

from loguru import logger
from sapien import Pose, Scene
from sapien.wrapper.articulation_builder import (
    ArticulationBuilder as SapienArticulationBuilder,
)
from sapien.wrapper.articulation_builder import JointRecord, LinkBuilder, MimicJointRecord
from sapien.wrapper.urdf_loader import URDFLoader as SapienURDFLoader

from ...configs.config import Config
from .pose import PoseConfig

if TYPE_CHECKING:
    pass


@dataclass
class JointConfig(Config):
    parent: str | None = None
    pose: PoseConfig = PoseConfig()  # .field()()


@dataclass
class CutConfig(Config):
    root: str
    cuts: list[str] = field(default_factory=list)


class URDFLoader(SapienURDFLoader):
    name: str = None
    disable_self_collisions: bool = False
    load_multiple_collisions_from_file: bool = False
    multiple_collisions_decomposition: str = "none"

    def parse_after_editing(
        self, urdf_file, srdf_file=None, package_dir=None, config: URDF = None
    ) -> SapienArticulationBuilder:
        articulation_builders, actor_builders, cameras = self.parse(urdf_file, srdf_file, package_dir)

        assert len(articulation_builders) == 1, "URDF contains multiple articulations"

        assert len(actor_builders) == 0, "URDF contains actors, which are not supported"
        assert len(cameras) == 0, "URDF contains cameras, which are not supported"

        articulation_builder = articulation_builders[0]

        # cut joints
        assert config is not None, "URDF config must be provided"
        if config.cut is not None:
            articulation_builder = self.cut_links(articulation_builder, config.cut)

        # fix joints
        if len(config.fixed_joints) > 0:
            articulation_builder = self.fix_joints(articulation_builder, config.fixed_joints)

        # add new link
        if len(config.new_links) > 0:
            articulation_builder = self.add_links(articulation_builder, config.new_links)

        # obtain mimic joints records
        if len(config.mimic_joints) > 0:
            for k, v in config.mimic_joints.items():
                articulation_builder.mimic_joint_records.append(
                    MimicJointRecord(
                        joint=k,
                        mimic=v["mimic"],
                        multiplier=v["multiplier"],
                        offset=v["offset"],
                    )
                )
        return articulation_builder

    @staticmethod
    def cut_links(articulation_builder: SapienArticulationBuilder, cut_config: CutConfig) -> SapienArticulationBuilder:
        root_link = next((lb for lb in articulation_builder.link_builders if lb.name == cut_config.root), None)
        if root_link is None:
            raise ValueError(f"Root link {cut_config.root} not found in articulation builder")

        cut_names = set(cut_config.cuts)
        visited: set[LinkBuilder] = set()

        def dfs(link: LinkBuilder):
            if link in visited or link.name in cut_names:
                return
            visited.add(link)
            # go up to parent if not in cut names
            if link.parent and link.parent.name not in cut_names:
                dfs(link.parent)
            # go down to children if not in cut names
            for child in articulation_builder.link_builders:
                if child.parent is link and child.name not in cut_names:
                    dfs(child)

        dfs(root_link)

        # Create a new articulation builder for the cut
        new_builder = articulation_builder.__class__()
        new_builder.set_scene(articulation_builder.scene)  # type: ignore
        new_builder.set_initial_pose(articulation_builder.initial_pose)

        # only copy mimic joint that's still valid
        new_builder.mimic_joint_records = []
        existing_joints = [i.joint_record.name for i in visited if i.joint_record is not None]
        for mimic in articulation_builder.mimic_joint_records:
            if mimic.joint in existing_joints:  # only copy mimic joints that are still valid
                new_builder.mimic_joint_records.append(copy.deepcopy(mimic))

        # Set the root link as the new root
        root = None
        for lb in visited:
            # print(lb.parent.name, lb.name)
            if lb.parent is None or lb.parent not in visited:
                root = lb
                if root.parent is not None:
                    root.parent = None
                    root.joint_record = copy.deepcopy(articulation_builder.link_builders[0].joint_record)
                    break
        if root is None:
            raise ValueError("Cannot find root link; circular dependency")

        # Append only visited links (in original order), update index
        for lb in articulation_builder.link_builders:
            if lb in visited:
                lb.index = len(new_builder.link_builders)
                new_builder.link_builders.append(lb)

        del articulation_builder  # Free memory of the old builder

        return new_builder

    @staticmethod
    def add_links(
        articulation_builder: SapienArticulationBuilder, link_config: dict[str, JointConfig]
    ) -> SapienArticulationBuilder:
        parents = [link_c.parent for link_c in link_config.values()]

        name_to_index = {lb.name: lb.index for lb in articulation_builder.link_builders}
        parent_index = [name_to_index.get(name, -1) for name in parents]

        # check if all parents are valid
        if -1 in parent_index:
            raise ValueError("Invalid parent link specified")

        # add links to articulation builder
        for i, (link_name, link_c) in enumerate(link_config.items()):
            new_link = LinkBuilder(
                len(articulation_builder.link_builders) + i, articulation_builder.link_builders[parent_index[i]]
            )
            new_link.name = link_name
            new_link.joint_record = JointRecord(
                joint_type="fixed",
                pose_in_parent=link_c.pose.to_sapien_pose(),
                pose_in_child=PoseConfig().to_sapien_pose(),  # Assuming no child
                name=link_name,
            )
            articulation_builder.link_builders.append(new_link)
        return articulation_builder

    @staticmethod
    def fix_joints(
        articulation_builder: SapienArticulationBuilder, fix_joint_config: dict[str, float]
    ) -> SapienArticulationBuilder:
        for joint_name, joint_value in fix_joint_config.items():
            joint_record = next(
                (lb.joint_record for lb in articulation_builder.link_builders if lb.joint_record.name == joint_name),
                None,
            )

            if joint_record is None:
                raise ValueError(f"Cannot find joint {joint_name}")

            joint_type = joint_record.joint_type

            if joint_type == "prismatic":
                assert len(joint_record.limits) == 1
                lower, upper = joint_record.limits[0]  # type: ignore
                assert lower <= joint_value <= upper
                joint_record.pose_in_child *= Pose([-joint_value, 0, 0])
                joint_record.joint_type = "fixed"
            elif joint_type == "fixed":
                logger.warning(f"{joint_name} has already been fixed, skip fixing it again.")

            elif joint_type.startswith("revolute"):
                if joint_type == "revolute_unwrapped":
                    assert joint_record.limits[0][0] <= joint_value <= joint_record.limits[0][1]
                import transforms3d

                quat = transforms3d.quaternions.axangle2quat([1.0, 0.0, 0.0], -joint_value)
                joint_record.pose_in_child *= Pose([0, 0, 0], quat)
                joint_record.joint_type = "fixed"

            else:
                raise ValueError(f"Unknown joint type {joint_type}")

        return articulation_builder


@dataclass
class URDF(Config):
    urdf: Path | None = None
    srdf: Path | None = None

    fix_root_link: bool = False

    qpos: List[float] = field(default_factory=list)
    cut: CutConfig | None = None
    fixed_joints: dict[str, float] = field(default_factory=dict)
    new_links: dict[str, JointRecord] = field(default_factory=dict)
    mimic_joints: dict[str, dict] = field(default_factory=dict)

    def load_builder_after_editing(self, scene: Scene) -> SapienArticulationBuilder:
        parser = URDFLoader()
        parser.set_scene(scene)

        assert self.urdf is not None, "URDF file must be specified"
        urdf_file = str(self.urdf.resolve())
        assert urdf_file.endswith(".urdf")
        srdf_file = str(self.srdf) if self.srdf else urdf_file[:-4] + "srdf"

        articulation_builder = parser.parse_after_editing(urdf_file, srdf_file, None, self)
        return articulation_builder

    def load(self, scene: Scene, name: str = None):
        builder = self.load_builder_after_editing(scene)
        robot = builder.build(fix_root_link=self.fix_root_link)

        robot.name = name
        if len(self.qpos) > 0:
            assert len(self.qpos) == len(robot.get_qpos()), "qpos length does not match articulation qpos length"

            robot.set_qpos(self.qpos)
        return robot
