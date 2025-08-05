import numpy as np
import numpy.typing as npt
import os
from lxml import etree

from transic_envs.envs.core.furniture.furniture_parts.base import Part
from transic_envs.utils.pose_utils import get_mat, rot_mat, is_similar_rot
from transic_envs.asset_root import ASSET_ROOT


class Leg(Part):
    def __init__(self, part_config, part_idx, seed):
        super().__init__(part_config, part_idx, seed)
        tag_ids = part_config["ids"]

        self.rel_pose_from_center[tag_ids[0]] = get_mat(
            [0, 0, -self.tag_offset], [0, 0, 0]
        )
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [-self.tag_offset, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat(
            [0, 0, self.tag_offset], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [self.tag_offset, 0, 0], [0, -np.pi / 2, 0]
        )

        self.done = False
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.25

        self.skill_complete_next_states = [
            "lift_up",
            "insert",
        ]  # Specificy next state after skill is complete. Screw done is handle in `get_assembly_action`

        self.reset()

        self.part_attached_skill_idx = 4

    def reset(self):
        self.prev_pose = None
        self._state = "reach_leg_floor_xy"
        self.gripper_action = -1

    def is_in_reset_ori(
        self, pose: npt.NDArray[np.float32], from_skill, ori_bound
    ) -> bool:
        # y-axis of the leg align with y-axis of the base.
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False


class SquareTableLeg(Leg):
    def __init__(self, part_config, part_idx, seed):
        self.tag_offset = 0.015
        self.half_width = 0.015

        self.reset_x_len = 0.03 + 0.02  # 0.02 is the margin
        self.reset_y_len = 0.0875
        super().__init__(part_config, part_idx, seed)

        self.reset_gripper_width = 0.06
        self.grasp_margin_x = 0
        self.grasp_margin_z = 0


class RandomCylinderLeg(Leg):
    def __init__(self, part_config, part_idx, seed):
        # This part is created procedurally, so we don't need the original asset file.
        # However, we'll use it as a template.
        template_asset_file = "random_cylinder/random_cylinder.urdf" 

        # Randomize cylinder properties
        self.random = np.random.RandomState(seed)
        self.length = self.random.uniform(0.1, 0.2)
        self.radius = self.random.uniform(0.01, 0.02)
        self.top_grap = self.length*0.3

        # Create a unique path for the modified URDF.
        temp_dir = os.path.join(ASSET_ROOT, "random_cylinder", f"temp_{part_idx}_{self.random.randint(0, 100000)}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_urdf_path = os.path.join(temp_dir, "random_cylinder.urdf")

        # Modify the URDF with the new dimensions
        urdf_path = os.path.join(ASSET_ROOT, template_asset_file)
        tree = etree.parse(urdf_path)
        root = tree.getroot()

        for cylinder in root.iter('cylinder'):
            cylinder.set('length', str(self.length))
            cylinder.set('radius', str(self.radius))

        tree.write(temp_urdf_path, pretty_print=True, xml_declaration=True, encoding="utf-8")

        # Update the asset file path in the part_config FOR THIS INSTANCE
        part_config['asset_file'] = os.path.relpath(temp_urdf_path, ASSET_ROOT)

        self.tag_offset = self.radius
        self.half_width = self.radius
        
        super().__init__(part_config, part_idx, seed)

        self.reset_x_len = self.radius * 2 + 0.02
        self.reset_y_len = self.length
        
        self.reset_gripper_width = self.radius * 2
        self.grasp_margin_x = 0
        self.grasp_margin_z = 0
