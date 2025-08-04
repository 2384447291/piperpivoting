from transic_envs.envs.core.furniture.base import Furniture
from transic_envs.envs.core.furniture.furniture_parts.base import Part
from transic_envs.asset_root import ASSET_ROOT
import os


class CylinderPart(Part):
    def __init__(self, part_config, part_idx, seed):
        super().__init__(part_config, part_idx, seed)
        self.name = "cylinder"


class RandomCylinder(Furniture):
    def __init__(self, seed):
        super().__init__(seed)
        
        # Define the part using a dictionary for clarity
        cylinder_config = {
            "name": "cylinder",
            "asset_file": os.path.join(ASSET_ROOT, "random_cylinder/cylinder.urdf"),
            "ids": [0], # Dummy tag ID
            "reset_pos": [[0, 0, 0.1]], # Initial position
            "reset_ori": [[1, 0, 0, 0, 1, 0, 0, 0, 1]] # Initial orientation (identity matrix)
        }

        self.parts = [CylinderPart(cylinder_config, 0, seed)]
        self.num_parts = len(self.parts)

