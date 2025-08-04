from __future__ import annotations

from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import numpy as np

from transic_envs.asset_root import ASSET_ROOT
from transic_envs.utils.pose_utils import get_mat
from transic_envs.envs.core.vec_task import VecTask
from transic_envs.envs.core.sim_config import sim_config
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core.furniture import furniture_factory
from transic_envs.envs.core.furniture.config import config, ROBOT_HEIGHT

#PiperEnvOSC(PiperEnvOSCBase) → PiperEnvOSCBase(VecTask) → VecTask

################################### A. 配置相关变量 ###################################
# self.cfg                    # 配置字典
# self.furniture              # 家具对象 (furniture_factory创建)
# self._num_fparts           # 家具部件数量


################################### B. 机器人相关变量 ###################################
# # 关节限制
# self.piper_dof_lower_limits    # 关节下限 [8,] tensor
# self.piper_dof_upper_limits    # 关节上限 [8,] tensor
# self.piper_dof_speed_scales    # 关节速度缩放 [8,] tensor

# # 机器人状态
# self.piper_handles            # 机器人部件句柄字典
# self.num_piper_bodies         # 机器人刚体数量
# self.num_piper_dofs          # 机器人自由度数量 (8)
# self.piper_grip_site_idx     # 末端执行器索引

# # 控制相关
# self._arm_control            # 机械臂控制 [num_envs, 6]
# self._gripper_control        # 夹爪控制 [num_envs, 2]
# self._pos_control           # 位置控制 [num_envs, 8]
# self._effort_control        # 力矩控制 [num_envs, 8]

################################### C. 状态变量 ###################################
# self.states                  # 状态字典，包含：
# # - q: 关节角度 [num_envs, 8]
# # - cos_q: 关节角度余弦 [num_envs, 8]  
# # - sin_q: 关节角度正弦 [num_envs, 8]
# # - dq: 关节角速度 [num_envs, 8]
# # - q_gripper: 夹爪关节角度 [num_envs, 2]
# # - eef_pos: 末端执行器位置 [num_envs, 3]
# # - eef_quat: 末端执行器姿态 [num_envs, 4]
# # - gripper_width: 夹爪宽度 [num_envs, 1]
# # - eef_lf_pos: 左手指位置 [num_envs, 3]
# # - eef_rf_pos: 右手指位置 [num_envs, 3]

# self._q                     # 关节角度 [num_envs, 8]
# self._qd                    # 关节角速度 [num_envs, 8]
# self._eef_state            # 末端执行器状态 [num_envs, 13]
# self._base_state           # 基座状态 [num_envs, 13]
# self._eef_lf_state         # 左手指状态 [num_envs, 13]
# self._eef_rf_state         # 右手指状态 [num_envs, 13]
# 13维状态向量的组成：
# [0:3]   # 位置 (x, y, z) - 世界坐标系中的位置
# [3:7]   # 旋转 (四元数 w, x, y, z) - 世界坐标系中的姿态
# [7:10]  # 线速度 (vx, vy, vz) - 世界坐标系中的线速度
# [10:13] # 角速度 (wx, wy, wz) - 世界坐标系中的角速度

################################### D. 环境相关变量 ###################################
# self.envs                   # 环境指针列表
# self.pipers                 # 机器人actor列表
# self.num_envs              # 环境数量
# self.num_dofs              # 总自由度数量
# self.actions               # 当前动作 [num_envs, num_actions]

################################### E. 观察和奖励变量 ###################################
# self.obs_dict              # 观察字典
# self.rew_buf              # 奖励缓冲区 [num_envs]
# self.reset_buf            # 重置缓冲区 [num_envs]
# self.progress_buf         # 进度缓冲区 [num_envs]
# self.max_episode_length   # 最大episode长

# PiperEnvOSCBase - 操作空间控制基类（Piper机械臂）
# 主要特点：
# 适配Piper机械臂（6+2自由度），操作空间控制
class PiperEnvOSCBase(VecTask):
    piper_asset_file = "piper_description/urdf/piper_description.urdf"

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
        num_furniture_parts: int = 0,
    ):
        self._record = record
        self.cfg = cfg

        #默认使用四元数旋转
        use_quat_rot = self.use_quat_rot = self.cfg["env"].get("useQuatRot", False)
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        #默认关节角速度噪声为0.01
        self.piper_dof_noise = self.cfg["env"].get("piperDofNoise", 0.01)
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        #这个base还没有实现家具部件，所以num_furniture_parts为0,家具的实现是在下面的PiperEnvOSC中
        self._num_fparts = num_furniture_parts

        self.cfg["env"]["numActions"] = 7 if use_quat_rot else 6

        self._prop_dump_info = self.cfg["env"].get("propDumpInfo", {})

        #在运行时，需要填充的变量
        self.states = {}
        self.piper_handles = {} # 将名字映射到相关的sim句柄
        self.fparts_handles = {} # 家具部件句柄
        self.num_dofs = None # 每个环境的总自由度数量
        self.actions = None # 当前要部署的动作
        self._fparts_names = [] # 所有家具部件的名称
        self._init_fparts_states = None # 所有家具部件的初始状态
        self._fparts_states = None # 所有家具部件的状态

        self._root_state = None
        self._dof_state = None
        self._q = None
        self._qd = None
        self._rigid_body_state = None
        self.net_cf = None
        self._eef_state = None
        self._ftip_center_state = None
        self._eef_lf_state = None
        self._eef_rf_state = None
        self._j_eef = None
        self._mm = None
        self._arm_control = None
        self._gripper_control = None
        self._pos_control = None
        self._effort_control = None
        self._piper_effort_limits = None
        self._global_piper_indices = None
        self._global_furniture_part_indices = {}

        self._front_wall_idxs = None
        self._left_wall_idxs = None
        self._right_wall_idxs = None
        self._fparts_idxs = None

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

        # Piper默认关节初始位姿（6+2）
        default_pose = self.cfg["env"].get("piperDefaultDofPos", None)
        default_pose = default_pose or [
            0.0, 1.5, -0.5, 0.0, -0.8, 0.0,  # 6个主关节
            0.05, -0.05                      # 2个夹爪关节
        ]
        self.piper_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)

        # OSC Gains
        self.kp = torch.tensor([150.0] * 6, device=self.sim_device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.tensor([10.0] * 6, device=self.sim_device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        self.cmd_limit = torch.tensor(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.sim_device
        ).unsqueeze(0)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _import_furniture_assets(self):
        pass

    def _import_piper_pcd(self):
        pass

    def _import_obstacle_pcds(self):
        pass

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.3, 0.8, 0.03, table_asset_options)

        # import front obstacle asset
        front_obstacle_asset_options = gymapi.AssetOptions()
        front_obstacle_asset_options.fix_base_link = True
        front_obstacle_asset_file = "furniture_bench/urdf/obstacle_front.urdf"
        front_obstacle_asset = self.gym.load_asset(
            self.sim,
            ASSET_ROOT,
            front_obstacle_asset_file,
            front_obstacle_asset_options,
        )

        # import side obstacle asset
        side_obstacle_asset_options = gymapi.AssetOptions()
        side_obstacle_asset_options.fix_base_link = True
        side_obstacle_asset_file = "furniture_bench/urdf/obstacle_side.urdf"
        side_obstacle_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, side_obstacle_asset_file, side_obstacle_asset_options
        )

        # import background if recording
        if self._record:
            bg_asset_options = gymapi.AssetOptions()
            bg_asset_options.fix_base_link = True
            background_asset_file = "furniture_bench/urdf/background.urdf"
            background_asset = self.gym.load_asset(
                self.sim, ASSET_ROOT, background_asset_file, bg_asset_options
            )

        # import obstacle pcds
        self._import_obstacle_pcds()

        # import furniture assets
        self._import_furniture_assets()

        # import piper pcds
        self._import_piper_pcd()

        # load piper asset
        piper_asset_file = self.piper_asset_file
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False 
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        piper_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, piper_asset_file, asset_options
        )
        piper_dof_stiffness = torch.tensor(
            [0, 0, 0, 0, 0, 0, 5000.0, 5000.0],
            dtype=torch.float,
            device=self.sim_device,
        )
        piper_dof_damping = torch.tensor(
            [0, 0, 0, 0, 0, 0, 500, 500],
            dtype=torch.float,
            device=self.sim_device,
        )

        self.num_piper_bodies = self.gym.get_asset_rigid_body_count(piper_asset)
        self.num_piper_dofs = self.gym.get_asset_dof_count(piper_asset)

        piper_link_dict = self.gym.get_asset_rigid_body_dict(piper_asset)
        self.piper_grip_site_idx = piper_link_dict["gripper_base"]

        print(f"Num Piper Bodies: {self.num_piper_bodies}")
        print(f"Num Piper DOFs: {self.num_piper_dofs}")

        # set piper dof properties
        piper_dof_props = self.gym.get_asset_dof_properties(piper_asset)
        self.piper_dof_lower_limits = []
        self.piper_dof_upper_limits = []
        self._piper_effort_limits = []
        for i in range(self.num_piper_dofs):
            piper_dof_props["driveMode"][i] = (
                gymapi.DOF_MODE_POS if i > 5 else gymapi.DOF_MODE_EFFORT
            )
            piper_dof_props["stiffness"][i] = piper_dof_stiffness[i]
            piper_dof_props["damping"][i] = piper_dof_damping[i]

            self.piper_dof_lower_limits.append(piper_dof_props["lower"][i])
            self.piper_dof_upper_limits.append(piper_dof_props["upper"][i])
            self._piper_effort_limits.append(piper_dof_props["effort"][i])

        self.piper_dof_lower_limits = torch.tensor(
            self.piper_dof_lower_limits, device=self.sim_device
        )
        self.piper_dof_upper_limits = torch.tensor(
            self.piper_dof_upper_limits, device=self.sim_device
        )
        self._piper_effort_limits = torch.tensor(
            self._piper_effort_limits, device=self.sim_device
        )
        self.piper_dof_speed_scales = torch.ones_like(self.piper_dof_lower_limits)
        self.piper_dof_speed_scales[[6, 7]] = 0.1
        piper_dof_props["effort"][6] = 20
        piper_dof_props["effort"][7] = 20

        # Define start pose for piper and table
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.piper_pose = gymapi.Transform()
        table_half_width = 0.015
        self._table_surface_z = table_surface_z = table_pos.z + table_half_width
        self.piper_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )

        # Define start pose for obstacles
        base_tag_pose = gymapi.Transform()
        base_tag_pos = get_mat(
            (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
        )[:3, 3]
        base_tag_pose.p = self.piper_pose.p + gymapi.Vec3(
            base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
        )
        base_tag_pose.p.z = table_surface_z
        self._front_obstacle_pose = gymapi.Transform()
        self._front_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015
        )
        self._front_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._right_obstacle_pose = gymapi.Transform()
        self._right_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            -0.175,
            table_surface_z + 0.015,
        )
        self._right_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._left_obstacle_pose = gymapi.Transform()
        self._left_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            0.175,
            table_surface_z + 0.015,
        )
        self._left_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )

        if self._record:
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)

        # compute aggregate size
        num_piper_bodies = self.gym.get_asset_rigid_body_count(piper_asset)
        num_piper_shapes = self.gym.get_asset_rigid_shape_count(piper_asset)
        max_agg_bodies = (
            num_piper_bodies
            + 4
            + self._num_fparts
            + (1 if self._record else 0)  # for background
        )  # 1 for table, front obstacle, left obstacle, right obstacle
        max_agg_shapes = (
            num_piper_shapes + 4 + self._num_fparts + (1 if self._record else 0)
        )

        self.pipers = []
        self.envs = []

        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: piper should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create piper
            piper_actor = self.gym.create_actor(
                env_ptr,
                piper_asset,
                self.piper_pose,
                "piper",
                i,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, piper_actor, piper_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0
            )
            table_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_actor
            )
            table_props[0].friction = 0.10
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(
                env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1)
            )
            front_obstacle_actor = self.gym.create_actor(
                env_ptr,
                front_obstacle_asset,
                self._front_obstacle_pose,
                "obstacle_front",
                i,
                0,
            )
            left_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._left_obstacle_pose,
                "obstacle_left",
                i,
                0,
            )
            right_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._right_obstacle_pose,
                "obstacle_right",
                i,
                0,
            )

            if self._record:
                bg_actor = self.gym.create_actor(
                    env_ptr,
                    background_asset,
                    bg_pose,
                    "background",
                    i,
                    0,
                )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create furniture parts
            self._create_furniture_parts(env_ptr, i)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.pipers.append(piper_actor)

        self.piper_from_origin_mat = get_mat(
            [self.piper_pose.p.x, self.piper_pose.p.y, self.piper_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
        april_to_sim_mat = self.piper_from_origin_mat @ self.base_tag_from_robot_mat
        self.april_to_sim_mat = torch.from_numpy(april_to_sim_mat).to(
            device=self.sim_device
        )

        # Setup init state buffer for all furniture parts
        self._init_fparts_states = {
            part_name: torch.zeros(self.num_envs, 13, device=self.sim_device)
            for part_name in self._fparts_names
        }

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        piper_handle = 0
        self.piper_handles = {
            # Piper
            "hand": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "gripper_base"
            ),
            "base": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "base_link"
            ),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "gripper_left"
            ),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "gripper_right"
            ),
            "leftfinger": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "gripper_left"
            ),
            "rightfinger": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "gripper_right"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "gripper_base"
            ),
            "fingertip_center": self.gym.find_actor_rigid_body_handle(
                env_ptr, piper_handle, "tri_finger_center"
            ),
        }
        self.fparts_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in self._fparts_names
        }
        self.walls_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in ["obstacle_front", "obstacle_left", "obstacle_right"]
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.piper_handles["grip_site"], :]
        self._ftip_center_state = self._rigid_body_state[
            :, self.piper_handles["fingertip_center"], :
        ]
        self._base_state = self._rigid_body_state[:, self.piper_handles["base"], :]
        self._eef_lf_state = self._rigid_body_state[
            :, self.piper_handles["leftfinger_tip"], :
        ]
        self._eef_rf_state = self._rigid_body_state[
            :, self.piper_handles["rightfinger_tip"], :
        ]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "piper")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._j_eef = jacobian[
            :, self.piper_grip_site_idx - 1, :, :6
        ]  # -1 due to fixed base link.
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "piper")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]
        self._fparts_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.fparts_handles.items()
        }
        self._walls_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.walls_handles.items()
        }
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf)

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :6]  
        self._gripper_control = self._pos_control[:, 6:8]

        # Initialize indices
        self._global_piper_indices = torch.tensor(
            [
                self.gym.find_actor_index(env, "piper", gymapi.DOMAIN_SIM)
                for env in self.envs
            ],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_furniture_part_indices = {
            part_name: torch.tensor(
                [
                    self.gym.find_actor_index(env, part_name, gymapi.DOMAIN_SIM)
                    for env in self.envs
                ],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
            for part_name in self._fparts_names
        }
        self._global_wall_indices = {
            part_name: torch.tensor(
                [
                    self.gym.find_actor_index(env, part_name, gymapi.DOMAIN_SIM)
                    for env in self.envs
                ],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
            for part_name in ["obstacle_front", "obstacle_left", "obstacle_right"]
        }

        # Initialize wall indices for contact force
        self._front_wall_idxs = None
        self._left_wall_idxs = None
        self._right_wall_idxs = None
        self._fparts_idxs = None

    def _create_furniture_parts(self, env_prt, i):
        pass

    def _update_states(self):
        self.states.update(
            {
                "q": self._q[:, :],
                "cos_q": torch.cos(self._q[:, :]),
                "sin_q": torch.sin(self._q[:, :]),
                "dq": self._qd[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3] - self._base_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "ftip_center_pos": self._ftip_center_state[:, :3] - self._base_state[:, :3],
                "ftip_center_quat": self._ftip_center_state[:, 3:7],
                "gripper_width": torch.abs(self._q[:, -2] - self._q[:, -1]).unsqueeze(-1),
                "eef_vel": self._eef_state[:, 7:],  # still required for OSC
                "eef_lf_pos": self._eef_lf_state[:, :3] - self._base_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3] - self._base_state[:, :3],
            }
        )

        fparts_states = {}
        for name, state in self._fparts_states.items():
            pos, rot, vel = (
                state[:, :3] - self._base_state[:, :3],
                state[:, 3:7],
                state[:, 7:13],
            )
            fparts_states[f"{name}_pos"] = pos
            fparts_states[f"{name}_rot"] = rot
            fparts_states[f"{name}_vel"] = vel
        self.states.update(fparts_states)
        walls_states = {}
        for name, state in self._walls_states.items():
            pos, rot = state[:, :3] - self._base_state[:, :3], state[:, 3:7]
            walls_states[f"{name}_pos"] = pos
        self.states.update(walls_states)

        if self._front_wall_idxs is None:
            front_wall_idxs, left_wall_idxs, right_wall_idxs = [], [], []
            for env_handle in self.envs:
                front_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_front"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                left_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_left"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                right_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_right"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
            self._front_wall_idxs = torch.tensor(
                front_wall_idxs, device=self.sim_device
            )
            self._left_wall_idxs = torch.tensor(left_wall_idxs, device=self.sim_device)
            self._right_wall_idxs = torch.tensor(
                right_wall_idxs, device=self.sim_device
            )

        if self._fparts_idxs is None:
            fparts_idxs = {name: [] for name in self._fparts_states.keys()}
            for env_handle in self.envs:
                for name, handle in self.fparts_handles.items():
                    fparts_idxs[name].append(
                        self.gym.get_actor_rigid_body_index(
                            env_handle,
                            handle,
                            0,
                            gymapi.DOMAIN_SIM,
                        )
                    )
            self._fparts_idxs = {
                k: torch.tensor(v, device=self.sim_device)
                for k, v in fparts_idxs.items()
            }

        self.states.update(
            {
                "front_wall_cf": self.net_cf[self._front_wall_idxs],
                "left_wall_cf": self.net_cf[self._left_wall_idxs],
                "right_wall_cf": self.net_cf[self._right_wall_idxs],
            }
        )
        self.states.update(
            {
                f"{name}_cf": self.net_cf[idxs]
                for name, idxs in self._fparts_idxs.items()
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._update_states()

    def _compute_osc_torques(self, dpose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[env_ids, :6], self._qd[env_ids, :6]
        mm = self._mm[env_ids]
        j_eef = self._j_eef[env_ids]
        mm_inv = torch.inverse(mm)
        m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(j_eef, 1, 2)
            @ m_eef
            @ (self.kp * dpose - self.kd * self.states["eef_vel"][env_ids]).unsqueeze(
                -1
            )
        )

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.piper_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null[:, 6:] *= 0
        u_null = mm @ u_null.unsqueeze(-1)
        u += (
            torch.eye(6, device=self.device).unsqueeze(0)
            - torch.transpose(j_eef, 1, 2) @ j_eef_inv
        ) @ u_null

        # Clip the values to be within valid effort range
        u = torch_jit_utils.tensor_clamp(
            u.squeeze(-1),
            -self._piper_effort_limits[:6].unsqueeze(0),
            self._piper_effort_limits[:6].unsqueeze(0),
        )
        return u

    def pre_physics_step(self, actions):
        if self.use_quat_rot:
            # 断言（检查）actions 最后一个维度必须是 7。
            # 意义：7 维动作向量包含 3 维位置、4 维四元数，无夹爪指令。
            assert (
                actions.shape[-1] == 7
            ), "Must provide 7D action for PiperPivotingQuatRot"
            pos, quat_rot = actions[:, :3], actions[:, 3:7]
            # rot_angle: (...,)
            # rot_axis: (..., 3)
            # 将四元数转换为旋转角度和旋转轴。
            # rot_angle：旋转角度 θ，形状与 quat_rot 第一个维度一致；
            # rot_axis：旋转轴方向向量，形状为 (batch, 3)。
            rot_angle, rot_axis = torch_jit_utils.quat_to_angle_axis(quat_rot)
            # get rotation along each axis
            rot = torch.stack([rot_angle * rot_axis[..., i] for i in range(3)], dim=-1)
            actions = torch.cat([pos, rot], dim=-1)
            # 把原来的 7 维动作重组为 6 维：
            # 3 维位置 + 3 维旋转，无夹爪指令。
            # 其中，旋转由旋转角度和旋转轴向量表示。
        else:
            # 非四元数模式：6维动作 (3位置 + 3旋转)
            assert (
                actions.shape[-1] == 6
            ), "Must provide 6D action for PiperPivoting"

        self.actions = actions.clone().to(self.device)

        # Only control arm, no gripper control
        u_arm = self.actions

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Set gripper to closed position (fixed)
        u_fingers = torch.zeros_like(self._gripper_control)
        # 设置夹爪为闭合状态（使用lower limits让夹爪闭合）
        u_fingers[:, 0] = self.piper_dof_lower_limits[-2].item()  # 左夹爪闭合
        u_fingers[:, 1] = self.piper_dof_upper_limits[-1].item()  # 右夹爪闭合
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers
        
        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        # 查找到reset_buf中为1的索引
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_dummy_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            self.max_episode_length,
        )

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(2.0, -0.00, 1.4)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def allocate_buffers(self):
        # 也为蒸馏等用途分配额外buffer
        super().allocate_buffers()
        # 基本属性字段
        self.dump_fileds = {
            k: torch.zeros(
                (self.num_envs, v),
                device=self.device,
                dtype=torch.float,
            )
            for k, v in self._prop_dump_info.items()
        }
        # 构建PCD时，仅保存动态部分（家具+机械臂手指）
        self.dump_fileds.update(
            {
                k: torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                )
                for k in self._fparts_states.keys()
            }
        )
        self.dump_fileds.update(
            {
                "leftfinger": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
                "rightfinger": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
                "piper_base": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
            }
        )

    def compute_observations(self):
        self._refresh()
        self.obs_dict["proprioception"][:] = torch.cat(
            [
                self.states[ob][:, :-2]
                if ob in ["q", "cos_q", "sin_q", "dq"]
                else self.states[ob]
                for ob in self._obs_keys
            ],
            dim=-1,
        )
        if len(self._privileged_obs_keys) > 0:
            self.obs_dict["privileged"][:] = torch.cat(
                [self.states[ob] for ob in self._privileged_obs_keys], dim=-1
            )
        # 更新dump字段
        for prop_name in self._prop_dump_info.keys():
            if prop_name in ["q", "cos_q", "sin_q", "dq"]:
                self.dump_fileds[prop_name][:] = self.states[prop_name][:, :-2]
            else:
                self.dump_fileds[prop_name][:] = self.states[prop_name][:]
        for fpart_name, fpart_state in self._fparts_states.items():
            self.dump_fileds[fpart_name][:] = fpart_state[:, :7]
        self.dump_fileds["leftfinger"][:] = self._rigid_body_state[
            :, self.piper_handles["leftfinger"], :
        ][:, :7]
        self.dump_fileds["rightfinger"][:] = self._rigid_body_state[
            :, self.piper_handles["rightfinger"], :
        ][:, :7]
        self.dump_fileds["piper_base"][:] = self._base_state[:, :7]
        return self.obs_dict

# PiperEnvOSC - 操作空间控制+家具
class PiperEnvOSC(PiperEnvOSCBase):
    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        furniture = cfg["env"]["furniture"]
        self.furniture = furniture_factory(furniture, cfg["seed"])
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
            num_furniture_parts=len(self.furniture.parts),
        )

    def _import_furniture_assets(self):
        self._fparts_assets = {}
        for part in self.furniture.parts:
            if part.name in self._fparts_assets:
                continue
            assert_option = sim_config["asset"][part.name]
            self._fparts_assets[part.name] = self.gym.load_asset(
                self.sim,
                ASSET_ROOT,
                part.asset_file,
                assert_option,
            )

    def _create_furniture_parts(self, env_prt, i):
        for part in self.furniture.parts:
            actor = self.gym.create_actor(
                env_prt,
                self._fparts_assets[part.name],
                gymapi.Transform(),
                part.name,
                i,
                0,
            )
            # Set properties of part
            part_props = self.gym.get_actor_rigid_shape_properties(env_prt, actor)
            part_props[0].friction = sim_config["parts"]["friction"]
            self.gym.set_actor_rigid_shape_properties(env_prt, actor, part_props)
            if part.name not in self._fparts_names:
                self._fparts_names.append(part.name)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        # Randomize initial furniture part poses
        self.furniture.reset()

        # Update furniture parts poses
        pos, ori = [], []
        for part in self.furniture.parts:
            pos.append(part.part_config["reset_pos"][0])  # (3,)
            ori.append(part.part_config["reset_ori"][0])  # (4,)
        pos = np.stack(pos)[:, np.newaxis, :]  # (num_parts, 1, 3)
        ori = np.stack(ori)[:, np.newaxis, ...]  # (num_parts, 1, 4, 4)
        pos = pos.repeat(len(env_ids), 1)  # (num_parts, num_resets, 3)
        ori = ori.repeat(len(env_ids), 1)  # (num_parts, num_resets, 4, 4)
        # randomize pos and ori
        pos[:, :, :2] += np.random.uniform(
            -0.015, 0.015, size=(len(self.furniture.parts), len(env_ids), 2)
        )
        pos = torch.tensor(pos, device=self.sim_device)
        # convert pos to homogenous matrix
        pos_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(len(self.furniture.parts), len(env_ids), 1, 1)
        )  # (num_parts, num_resets, 4, 4)
        pos_mat[:, :, :3, 3] = pos
        pos_mat = pos_mat.reshape(-1, 4, 4)
        pos_mat = (
            self.april_to_sim_mat @ pos_mat
        )  # (4, 4) @ (num_parts * num_resets, 4, 4) -> (num_parts * num_resets, 4, 4)
        pos_mat = pos_mat.reshape(len(self.furniture.parts), len(env_ids), 4, 4)
        new_pos = pos_mat[:, :, :3, 3]  # (num_parts, num_resets, 3)

        ori = torch.tensor(ori, device=self.sim_device)  # (num_parts, num_resets, 4, 4)
        ori_noise = np.zeros((len(self.furniture.parts), len(env_ids), 3))
        ori_noise[:, :, 2] = np.random.uniform(
            np.radians(-15),
            np.radians(15),
            size=(len(self.furniture.parts), len(env_ids)),
        )
        ori_noise = torch.tensor(ori_noise, device=self.sim_device, dtype=ori.dtype)
        ori_noise = torch_jit_utils.axisangle2quat(
            ori_noise
        )  # (num_parts, num_resets, 4) in xyzw order
        # change to wxyz order
        ori_noise = torch.cat([ori_noise[:, :, 3:], ori_noise[:, :, :3]], dim=-1)
        ori_noise = torch_jit_utils.quaternion_to_matrix(
            ori_noise
        )  # (num_parts, num_resets, 3, 3)
        # convert to homogeneous matrix
        ori_noise_homo = (
            torch.eye(4, dtype=ori.dtype, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(len(self.furniture.parts), len(env_ids), 1, 1)
        )  # (num_parts, num_resets, 4, 4)
        ori_noise_homo[:, :, :3, :3] = ori_noise
        ori_noise_homo[:, :, 3, 3] = 1
        ori = ori.reshape(-1, 4, 4)
        ori_noise_homo = ori_noise_homo.reshape(-1, 4, 4)
        ori = ori_noise_homo @ ori  # (N, 4, 4) @ (N, 4, 4) -> (N, 4, 4)
        ori = (
            self.april_to_sim_mat @ ori
        )  # (4, 4) @ (num_parts * num_resets, 4, 4) -> (num_parts * num_resets, 4, 4)
        ori_quat = torch_jit_utils.matrix_to_quaternion(
            ori[:, :3, :3]
        )  # (num_parts * num_resets, 4) in wxyz order
        # convert to xyzw order
        ori_quat = torch.cat([ori_quat[:, 1:], ori_quat[:, :1]], dim=-1)
        ori_quat = ori_quat.reshape(len(self.furniture.parts), len(env_ids), 4)

        reset_pos = torch.cat([new_pos, ori_quat], dim=-1)  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pos.dtype,
        )
        reset_state = torch.cat([reset_pos, vel], dim=-1)  # (num_parts, num_resets, 13)

        for part, part_state in zip(self.furniture.parts, reset_state):
            # Set furniture part state
            self._init_fparts_states[part.name][env_ids, :] = part_state
            # Write these new init states to the sim states
            self._fparts_states[part.name][env_ids] = self._init_fparts_states[
                part.name
            ][env_ids]
        # Collect all part ids and deploy state update
        multi_env_ids_int32 = [
            self._global_furniture_part_indices[part_name][env_ids].flatten()
            for part_name in self._fparts_names
        ]
        multi_env_ids_int32 = torch.cat(multi_env_ids_int32, dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

# PiperEnvJPCBase - 关节空间控制基类（Piper机械臂）
class PiperEnvJPCBase(VecTask):
    piper_asset_file = "piper_description/urdf/piper_description.urdf"

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        num_furniture_parts: int = 0,
    ):
        self._record = record
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.piper_dof_noise = self.cfg["env"].get("piperDofNoise", 0.01)
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self._num_fparts = num_furniture_parts
        self.cfg["env"]["numActions"] = 8
        self.states = {}
        self.piper_handles = {}
        self.fparts_handles = {}
        self.num_dofs = None
        self.actions = None
        self._fparts_names = []
        self._init_fparts_states = None
        self._fparts_states = None
        self._root_state = None
        self._dof_state = None
        self._q = None
        self._qd = None
        self._rigid_body_state = None
        self.net_cf = None
        self._eef_state = None
        self._ftip_center_state = None
        self._eef_lf_state = None
        self._eef_rf_state = None
        self._j_eef = None
        self._arm_control = None
        self._gripper_control = None
        self._pos_control = None
        self._global_piper_indices = None
        self._global_furniture_part_indices = {}
        self._front_wall_idxs = None
        self._left_wall_idxs = None
        self._right_wall_idxs = None
        self._fparts_idxs = None
        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )
        default_pose = self.cfg["env"].get("piperDefaultDofPos", None)
        default_pose = default_pose or [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 6个主关节
            0.0, 0.0                      # 2个夹爪关节
        ]
        self.piper_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)
        self.cmd_limit_high = torch.tensor(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.sim_device
        ).unsqueeze(0)
        self.cmd_limit_low = -self.cmd_limit_high
        damping = 0.05
        self._ik_lambda = torch.eye(6, device=self.sim_device) * (damping**2)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _import_furniture_assets(self):
        pass

    def _import_piper_pcd(self):
        pass

    def _import_obstacle_pcds(self):
        pass

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.3, 0.8, 0.03, table_asset_options)

        # import front obstacle asset
        front_obstacle_asset_options = gymapi.AssetOptions()
        front_obstacle_asset_options.fix_base_link = True
        front_obstacle_asset_file = "furniture_bench/urdf/obstacle_front.urdf"
        front_obstacle_asset = self.gym.load_asset(
            self.sim,
            ASSET_ROOT,
            front_obstacle_asset_file,
            front_obstacle_asset_options,
        )

        # import side obstacle asset
        side_obstacle_asset_options = gymapi.AssetOptions()
        side_obstacle_asset_options.fix_base_link = True
        side_obstacle_asset_file = "furniture_bench/urdf/obstacle_side.urdf"
        side_obstacle_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, side_obstacle_asset_file, side_obstacle_asset_options
        )

        # import background if recording
        if self._record:
            bg_asset_options = gymapi.AssetOptions()
            bg_asset_options.fix_base_link = True
            background_asset_file = "furniture_bench/urdf/background.urdf"
            background_asset = self.gym.load_asset(
                self.sim, ASSET_ROOT, background_asset_file, bg_asset_options
            )

        # import obstacle pcds
        self._import_obstacle_pcds()

        # import furniture assets
        self._import_furniture_assets()

        # import piper pcds
        self._import_piper_pcd()

        # load piper asset
        piper_asset_file = self.piper_asset_file
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False  # 修复视觉模型错位问题
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        piper_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, piper_asset_file, asset_options
        )
        piper_dof_stiffness = torch.tensor(
            [400, 400, 400, 400, 400, 400, 5000.0, 5000.0],
            dtype=torch.float,
            device=self.sim_device,
        )
        piper_dof_damping = torch.tensor(
            [80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2],
            dtype=torch.float,
            device=self.sim_device,
        )
        piper_dof_effort = torch.tensor(
            [200, 200, 200, 200, 200, 200, 200, 200],
            dtype=torch.float,
            device=self.sim_device,
        )

        self.num_piper_bodies = self.gym.get_asset_rigid_body_count(piper_asset)
        self.num_piper_dofs = self.gym.get_asset_dof_count(piper_asset)

        piper_link_dict = self.gym.get_asset_rigid_body_dict(piper_asset)
        self.piper_grip_site_idx = piper_link_dict["gripper_base"]

        print(f"Num Piper Bodies: {self.num_piper_bodies}")
        print(f"Num Piper DOFs: {self.num_piper_dofs}")

        # set piper dof properties
        piper_dof_props = self.gym.get_asset_dof_properties(piper_asset)
        self.piper_dof_lower_limits = []
        self.piper_dof_upper_limits = []
        for i in range(self.num_piper_dofs):
            piper_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            piper_dof_props["stiffness"][i] = piper_dof_stiffness[i]
            piper_dof_props["damping"][i] = piper_dof_damping[i]
            piper_dof_props["effort"][i] = piper_dof_effort[i]
            self.piper_dof_lower_limits.append(piper_dof_props["lower"][i])
            self.piper_dof_upper_limits.append(piper_dof_props["upper"][i])
        self.piper_dof_lower_limits = torch.tensor(
            self.piper_dof_lower_limits, device=self.sim_device
        )
        self.piper_dof_upper_limits = torch.tensor(
            self.piper_dof_upper_limits, device=self.sim_device
        )
        self.piper_dof_speed_scales = torch.ones_like(self.piper_dof_lower_limits)
        self.piper_dof_speed_scales[[6, 7]] = 0.1

        # Define start pose for piper and table
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.piper_pose = gymapi.Transform()
        table_half_width = 0.015
        self._table_surface_z = table_surface_z = table_pos.z + table_half_width
        self.piper_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )

        # Define start pose for obstacles
        base_tag_pose = gymapi.Transform()
        base_tag_pos = get_mat(
            (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
        )[:3, 3]
        base_tag_pose.p = self.piper_pose.p + gymapi.Vec3(
            base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
        )
        base_tag_pose.p.z = table_surface_z
        self._front_obstacle_pose = gymapi.Transform()
        self._front_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015
        )
        self._front_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._right_obstacle_pose = gymapi.Transform()
        self._right_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            -0.175,
            table_surface_z + 0.015,
        )
        self._right_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._left_obstacle_pose = gymapi.Transform()
        self._left_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            0.175,
            table_surface_z + 0.015,
        )
        self._left_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )

        if self._record:
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)

        # compute aggregate size
        num_piper_bodies = self.gym.get_asset_rigid_body_count(piper_asset)
        num_piper_shapes = self.gym.get_asset_rigid_shape_count(piper_asset)
        max_agg_bodies = (
            num_piper_bodies
            + 4
            + self._num_fparts
            + (1 if self._record else 0)  # for background
        )  # 1 for table, front obstacle, left obstacle, right obstacle
        max_agg_shapes = (
            num_piper_shapes + 4 + self._num_fparts + (1 if self._record else 0)
        )

        self.pipers = []
        self.envs = []

        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: piper should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create piper
            piper_actor = self.gym.create_actor(
                env_ptr,
                piper_asset,
                self.piper_pose,
                "piper",
                i,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, piper_actor, piper_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0
            )
            table_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_actor
            )
            table_props[0].friction = 0.10
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(
                env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1)
            )
            front_obstacle_actor = self.gym.create_actor(
                env_ptr,
                front_obstacle_asset,
                self._front_obstacle_pose,
                "obstacle_front",
                i,
                0,
            )
            left_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._left_obstacle_pose,
                "obstacle_left",
                i,
                0,
            )
            right_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._right_obstacle_pose,
                "obstacle_right",
                i,
                0,
            )

            if self._record:
                bg_actor = self.gym.create_actor(
                    env_ptr,
                    background_asset,
                    bg_pose,
                    "background",
                    i,
                    0,
                )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create furniture parts
            self._create_furniture_parts(env_ptr, i)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.pipers.append(piper_actor)

        self.piper_from_origin_mat = get_mat(
            [self.piper_pose.p.x, self.piper_pose.p.y, self.piper_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
        april_to_sim_mat = self.piper_from_origin_mat @ self.base_tag_from_robot_mat
        self.april_to_sim_mat = torch.tensor(april_to_sim_mat, device=self.sim_device)

        # Setup init state buffer for all furniture parts
        self._init_fparts_states = {
            part_name: torch.zeros(self.num_envs, 13, device=self.sim_device)
            for part_name in self._fparts_names
        }

        # Setup data
        self.init_data()

    def init_data(self):
        env_ptr = self.envs[0]
        piper_handle = 0
        self.piper_handles = {
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, piper_handle, "gripper_base"),
            "base": self.gym.find_actor_rigid_body_handle(env_ptr, piper_handle, "link1"),
            "leftfinger": self.gym.find_actor_rigid_body_handle(env_ptr, piper_handle, "gripper_left"),
            "rightfinger": self.gym.find_actor_rigid_body_handle(env_ptr, piper_handle, "gripper_right"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, piper_handle, "gripper_base"),
        }
        self.fparts_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in self._fparts_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.piper_handles["grip_site"], :]
        self._base_state = self._rigid_body_state[:, self.piper_handles["base"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.piper_handles["leftfinger"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.piper_handles["rightfinger"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "piper")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._j_eef = jacobian[:, self.piper_grip_site_idx - 1, :, :6]
        self._fparts_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.fparts_handles.items()
        }
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf)
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self._arm_control = self._pos_control[:, :6]
        self._gripper_control = self._pos_control[:, 6:8]
        self._global_piper_indices = torch.tensor(
            [
                self.gym.find_actor_index(env, "piper", gymapi.DOMAIN_SIM)
                for env in self.envs
            ],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_furniture_part_indices = {
            part_name: torch.tensor(
                [
                    self.gym.find_actor_index(env, part_name, gymapi.DOMAIN_SIM)
                    for env in self.envs
                ],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
            for part_name in self._fparts_names
        }

    def _create_furniture_parts(self, env_prt, i):
        for part in self.furniture.parts:
            actor = self.gym.create_actor(
                env_prt,
                self._fparts_assets[part.name],
                gymapi.Transform(),
                part.name,
                i,
                0,
            )
            # Set properties of part
            part_props = self.gym.get_actor_rigid_shape_properties(env_prt, actor)
            part_props[0].friction = sim_config["parts"]["friction"]
            self.gym.set_actor_rigid_shape_properties(env_prt, actor, part_props)
            if part.name not in self._fparts_names:
                self._fparts_names.append(part.name)

    def _update_states(self):
        self.states.update(
            {
                "q": self._q[:, :],
                "cos_q": torch.cos(self._q[:, :]),
                "sin_q": torch.sin(self._q[:, :]),
                "dq": self._qd[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3] - self._base_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "ftip_center_pos": self._ftip_center_state[:, :3]
                - self._base_state[:, :3],
                "ftip_center_quat": self._ftip_center_state[:, 3:7],
                "gripper_width": torch.sum(self._q[:, -2:], dim=-1, keepdim=True),
                "eef_vel": self._eef_state[:, 7:],  # still required for OSC
                "eef_lf_pos": self._eef_lf_state[:, :3] - self._base_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3] - self._base_state[:, :3],
            }
        )

        fparts_states = {}
        for name, state in self._fparts_states.items():
            pos, rot, vel = (
                state[:, :3] - self._base_state[:, :3],
                state[:, 3:7],
                state[:, 7:13],
            )
            fparts_states[f"{name}_pos"] = pos
            fparts_states[f"{name}_rot"] = rot
            fparts_states[f"{name}_vel"] = vel
        self.states.update(fparts_states)
        walls_states = {}
        for name, state in self._walls_states.items():
            pos, rot = state[:, :3] - self._base_state[:, :3], state[:, 3:7]
            walls_states[f"{name}_pos"] = pos
        self.states.update(walls_states)

        if self._front_wall_idxs is None:
            front_wall_idxs, left_wall_idxs, right_wall_idxs = [], [], []
            for env_handle in self.envs:
                front_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_front"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                left_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_left"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                right_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_right"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
            self._front_wall_idxs = torch.tensor(
                front_wall_idxs, device=self.sim_device
            )
            self._left_wall_idxs = torch.tensor(left_wall_idxs, device=self.sim_device)
            self._right_wall_idxs = torch.tensor(
                right_wall_idxs, device=self.sim_device
            )

        if self._fparts_idxs is None:
            fparts_idxs = {name: [] for name in self._fparts_states.keys()}
            for env_handle in self.envs:
                for name, handle in self.fparts_handles.items():
                    fparts_idxs[name].append(
                        self.gym.get_actor_rigid_body_index(
                            env_handle,
                            handle,
                            0,
                            gymapi.DOMAIN_SIM,
                        )
                    )
            self._fparts_idxs = {
                k: torch.tensor(v, device=self.sim_device)
                for k, v in fparts_idxs.items()
            }

        self.states.update(
            {
                "front_wall_cf": self.net_cf[self._front_wall_idxs],
                "left_wall_cf": self.net_cf[self._left_wall_idxs],
                "right_wall_cf": self.net_cf[self._right_wall_idxs],
            }
        )
        self.states.update(
            {
                f"{name}_cf": self.net_cf[idxs]
                for name, idxs in self._fparts_idxs.items()
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._update_states()

    def _compute_ik(self, dpose):
        j_eef_T = torch.transpose(self._j_eef, 1, 2)
        u = (
            j_eef_T
            @ torch.inverse(self._j_eef @ j_eef_T + self._ik_lambda)
            @ dpose.unsqueeze(-1)
        ).view(self.num_envs, -1)
        return u

    def step(self, actions):
        """
        Will internally invoke JPC controller N times until achieve the desired joint position
        Here `actions` specify desired joint position
        """
        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        goal_q, gripper_actions = self.actions[:, :-1], self.actions[:, -1]

        # invoke JPC controller N times until achieve the desired joint position
        for _ in range(10):
            arm_control_target = goal_q[:]
            arm_control_target = torch.clamp(
                arm_control_target,
                min=self.piper_dof_lower_limits[:6],
                max=self.piper_dof_upper_limits[:6],
            )
            self._arm_control[:, :] = arm_control_target[:, :]

            # gripper control
            u_fingers = torch.zeros_like(self._gripper_control)
            u_fingers[:, 0] = torch.where(
                gripper_actions >= 0.0,
                self.piper_dof_upper_limits[-2].item(),
                self.piper_dof_lower_limits[-2].item(),
            )
            u_fingers[:, 1] = torch.where(
                gripper_actions >= 0.0,
                self.piper_dof_lower_limits[-1].item(),
                self.piper_dof_upper_limits[-1].item(),
            )
            # Write gripper command to appropriate tensor buffer
            self._gripper_control[:, :] = u_fingers

            # Deploy actions
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self._pos_control)
            )

            # step physics and render each frame
            for i in range(self.control_freq_inv):
                self.gym.simulate(self.sim)

            if self.camera_obs is not None:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)

            if self.camera_obs is not None:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

            # post physics
            self.compute_observations()

            if self.camera_obs is not None:
                self.gym.end_access_image_tensors(self.sim)

            if self._rgb_viewr_renderer is not None:
                self.render()

            # break when all joints are within 1e-3 of desired joint position
            if torch.all(
                torch.max(
                    torch.abs(self.states["q"][:, :6] - goal_q),
                    dim=-1,
                )[0]
                < 1e-3
            ):
                break

        # now update buffer
        self.progress_buf += 1
        self.randomize_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_reward(self.actions)

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(
                self.obs_buf, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_dummy_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            actions=actions,
            states=self.states,
            max_episode_length=self.max_episode_length,
        )

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def allocate_buffers(self):
        # 也为蒸馏等用途分配额外buffer
        super().allocate_buffers()
        # 基本属性字段
        self.dump_fileds = {
            k: torch.zeros(
                (self.num_envs, v),
                device=self.device,
                dtype=torch.float,
            )
            for k, v in self._prop_dump_info.items()
        }
        # 构建PCD时，仅保存动态部分（家具+机械臂手指）
        self.dump_fileds.update(
            {
                k: torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                )
                for k in self._fparts_states.keys()
            }
        )
        self.dump_fileds.update(
            {
                "leftfinger": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
                "rightfinger": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
                "piper_base": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
            }
        )

    def compute_observations(self):
        self._refresh()
        self.obs_dict["proprioception"][:] = torch.cat(
            [
                self.states[ob][:, :-2]
                if ob in ["q", "cos_q", "sin_q", "dq"]
                else self.states[ob]
                for ob in self._obs_keys
            ],
            dim=-1,
        )
        if len(self._privileged_obs_keys) > 0:
            self.obs_dict["privileged"][:] = torch.cat(
                [self.states[ob] for ob in self._privileged_obs_keys], dim=-1
            )
        # 更新dump字段
        for prop_name in self._prop_dump_info.keys():
            if prop_name in ["q", "cos_q", "sin_q", "dq"]:
                self.dump_fileds[prop_name][:] = self.states[prop_name][:, :-2]
            else:
                self.dump_fileds[prop_name][:] = self.states[prop_name][:]
        for fpart_name, fpart_state in self._fparts_states.items():
            self.dump_fileds[fpart_name][:] = fpart_state[:, :7]
        self.dump_fileds["leftfinger"][:] = self._rigid_body_state[
            :, self.piper_handles["leftfinger"], :
        ][:, :7]
        self.dump_fileds["rightfinger"][:] = self._rigid_body_state[
            :, self.piper_handles["rightfinger"], :
        ][:, :7]
        self.dump_fileds["piper_base"][:] = self._base_state[:, :7]
        return self.obs_dict

# PiperEnvJPC - 关节空间控制+家具
class PiperEnvJPC(PiperEnvJPCBase):
    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
    ):
        furniture = cfg["env"]["furniture"]
        self.furniture = furniture_factory(furniture, cfg["seed"])
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            num_furniture_parts=len(self.furniture.parts),
        )

    def _import_furniture_assets(self):
        self._fparts_assets = {}
        for part in self.furniture.parts:
            if part.name in self._fparts_assets:
                continue
            assert_option = sim_config["asset"][part.name]
            self._fparts_assets[part.name] = self.gym.load_asset(
                self.sim,
                ASSET_ROOT,
                part.asset_file,
                assert_option,
            )

    def _create_furniture_parts(self, env_prt, i):
        for part in self.furniture.parts:
            actor = self.gym.create_actor(
                env_prt,
                self._fparts_assets[part.name],
                gymapi.Transform(),
                part.name,
                i,
                0,
            )
            # Set properties of part
            part_props = self.gym.get_actor_rigid_shape_properties(env_prt, actor)
            part_props[0].friction = sim_config["parts"]["friction"]
            self.gym.set_actor_rigid_shape_properties(env_prt, actor, part_props)
            if part.name not in self._fparts_names:
                self._fparts_names.append(part.name)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.furniture.reset()
        # 省略家具部件reset，参考base.py

#虚拟奖励函数，没有任何意义，只是为了满足抽象方法，真正的奖励函数在子类中实现
@torch.jit.script
def compute_dummy_reward(reset_buf, progress_buf, actions, states, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], float) -> Tuple[Tensor, Tensor]

    # dummy rewards
    rewards = torch.zeros_like(reset_buf)
    reset_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return rewards, reset_buf


    
