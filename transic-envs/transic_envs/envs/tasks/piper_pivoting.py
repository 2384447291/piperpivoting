from __future__ import annotations

from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import numpy as np
import os

from transic_envs.asset_root import ASSET_ROOT
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core import PiperEnvOSC
import numpy as np


class PiperPivotingEnv(PiperEnvOSC):
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
        #reward每一项的具体参数权重
        self._angle_reward = cfg["env"]["angleReward"]
        self._success_reward = cfg["env"]["successReward"]
        self._qd_penalty = cfg["env"]["qdPenalty"]
        self._action_penalty = cfg["env"]["actionPenalty"]
        self._ori_error_threshold = cfg["env"]["oriErrorThreshold"]

        # 初始化初始俯仰差缓存，在 super().__init__ 调用 reset_idx 之前就要先占位
        self._init_pitch_difference = None  # 将在 reset_idx 首次被真正分配
        
        # 存储初始计算的坐标系信息（改为存储矩阵形式）
        self._init_coordinate_frames = None  # 将在 reset_idx 首次被真正分配为矩阵形式
        
        # 存储每个环境中腿的尺寸 [radius, length]
        self.leg_dimensions = None

        self.np_random = np.random.RandomState(cfg["seed"])


        #导入正向运动学的包
        try:
            import casadi
            import urdf_parser_py
        except ImportError:
            raise ImportError(
                "Packages `casadi` and `urdf-parser-py` are required for the env `InsertSinglePCD`. Install them with `pip install casadi urdf-parser-py`."
            )
        from transic_envs.utils.urdf2casadi import URDFparser

        piper_parser = URDFparser()
        piper_parser.from_file(
            os.path.join(
                ASSET_ROOT,
                "piper_description/urdf/piper_description.urdf",
            )
        )
        self._ftip_center_fk_fn = piper_parser.get_forward_kinematics(
            root="base_link", tip="tri_finger_center"
        )["T_fk"]

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )


    #重置环境,输入的是需要重置的env_ids
    def reset_idx(self, env_ids):
        # 如果尚未为初始俯仰差分配张量（第一次调用时），现在根据当前 num_envs 分配
        if self._init_pitch_difference is None or self._init_pitch_difference.shape[0] != self.num_envs:
            self._init_pitch_difference = torch.zeros(self.num_envs, device=self.sim_device)
        # 存储初始计算的坐标系信息（矩阵形式：位置3 + 旋转矩阵9）
        if self._init_coordinate_frames is None or self._init_coordinate_frames.shape[0] != self.num_envs:
            self._init_coordinate_frames = torch.zeros(self.num_envs, 12, device=self.sim_device)  # pos(3) + rot_matrix(9)

        # 如果全局开启了随机化（self.randomize == True），则调用 apply_randomizations 方法，对环境中的参数（例如光照、摩擦系数等）应用域随机化（Domain Randomization）设定。
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)
        # 有多少的关节就初始化多少个关节的噪声
        reset_noise = torch.rand((len(env_ids), 8), device=self.sim_device)
        pos = torch_jit_utils.tensor_clamp(
            self.piper_default_dof_pos.unsqueeze(0)
            + self.piper_dof_noise * 2.0 * (reset_noise - 0.5),
            self.piper_dof_lower_limits.unsqueeze(0),
            self.piper_dof_upper_limits,
        )

        # 夹爪瞬间张开
        pos[:, -2:] = 0.25*torch.tensor([self.piper_dof_upper_limits[-2].item(),self.piper_dof_lower_limits[-1].item()], device=self.sim_device)

        # 重置内部观测位置和速度
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # 设置任何位置控制为当前位置，任何速度/力控制为0
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # 把具体的参数设置到gym的sim中
        multi_env_ids_int32 = self._global_piper_indices[env_ids].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        # 继续执行重置逻辑的其余部分
        self.progress_buf[env_ids] = 0  # 重置进度缓冲区
        self.reset_buf[env_ids] = 0    # 重置重置缓冲区
        self.success_buf[env_ids] = 0  # 重置成功缓冲区
        self.failure_buf[env_ids] = 0  # 重置失败缓冲区

        # 获取夹爪中心位置
        qs = self._q[env_ids, :6]  # 取前6个关节（不含手爪）
        ftip_center_poses = []
        for q in qs:
            q = list(q.cpu().numpy())
            ftip_center_poses.append(
                np.array(self._ftip_center_fk_fn(q)).astype(np.float32).reshape(4, 4)
            )
        ftip_center_poses = torch.tensor(
            np.stack(ftip_center_poses),
            device=self.sim_device,
            dtype=torch.float32,
        )  # (N, 4, 4)

        # 更新被操作物件（"leg"）的状态。有多少个env有多少个leg
        num_resets = len(env_ids)

        # 构造一个单位矩阵
        leg2ftip_center_transform = torch.eye(
            4, device=self.sim_device, dtype=torch.float32
        )  # (4, 4)

        leg2ftip_center_transform[0, 3] = 0.01
        leg2ftip_center_transform[1, 3] = 0
        leg2ftip_center_transform[2, 3] = -self.furniture.parts[0].top_grap

        leg2ftip_center_transform = leg2ftip_center_transform.unsqueeze(0).repeat(
            num_resets, 1, 1
        )

        leg_pose = leg2ftip_center_transform @ ftip_center_poses  # (N, 4, 4)
        leg_pos = leg_pose[:, :3, 3] + self._base_state[env_ids, :3]
        
        # 将初始位置和旋转矩阵存储起来（直接存储矩阵，避免四元数转换）
        leg_rot_matrix = leg_pose[:, :3, :3]  # (N, 3, 3)
        self._init_coordinate_frames[env_ids, :3] = leg_pos
        self._init_coordinate_frames[env_ids, 3:12] = leg_rot_matrix.reshape(len(env_ids), 9)

        random_angle_deg = torch.rand(num_resets, device=self.sim_device) * 45
        random_angle = random_angle_deg * np.pi / 180.0  # 转为弧度保存

        self._init_pitch_difference[env_ids] = 90 - random_angle_deg

        # 设置桌腿的旋转
        leg2ftip_center_rotation = torch_jit_utils.quat_from_euler_xyz(
            roll=torch.ones((1,), device=self.sim_device, dtype=torch.float32) * (np.pi/2),
            pitch=torch.ones((1,), device=self.sim_device, dtype=torch.float32) *  random_angle,
            yaw=torch.ones((1,), device=self.sim_device, dtype=torch.float32) * 0,
        )  # (N, 4) in xyzw order
        # 改为wxyz顺序
        leg2ftip_center_rotation = torch.cat(
            [leg2ftip_center_rotation[:, -1:], leg2ftip_center_rotation[:, :-1]], dim=-1
        )
        leg2ftip_center_rotation = torch_jit_utils.quaternion_to_matrix(
            leg2ftip_center_rotation
        )  # (N, 3, 3)
        leg_rot = leg2ftip_center_rotation @ leg_pose[:, :3, :3]  # (N, 3, 3)
        leg_rot = torch_jit_utils.matrix_to_quaternion(leg_rot)  # (N, 4) in wxyz order
        # 改为xyzw顺序
        leg_rot = torch.cat([leg_rot[:, -1:], leg_rot[:, :-1]], dim=-1)

        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_leg_state[:, :3] = leg_pos
        sampled_leg_state[:, 3:7] = leg_rot

        # 设置桌腿状态
        self._init_fparts_states["leg"][env_ids, :] = sampled_leg_state

        # 将这些新的初始状态写入仿真状态
        self._fparts_states["leg"][env_ids] = self._init_fparts_states["leg"][env_ids]

        # 部署桌腿状态更新
        multi_env_ids_leg_int32 = self._global_furniture_part_indices["leg"][env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_leg_int32),
            len(multi_env_ids_leg_int32),
        )

    # def post_physics_step(self):
    #     """重写post_physics_step方法，添加持续的可视化"""
    #     # 调用父类的方法
    #     super().post_physics_step()
        
    #     # 在仿真过程中持续显示坐标系和pitch向量
    #     if hasattr(self, 'viewer') and self.viewer is not None:
    #         # 获取当前所有环境的状态
    #         all_env_ids = torch.arange(self.num_envs, device=self.sim_device)
    #         self._visualize_coordinate_frame_continuous(all_env_ids)


    def _visualize_coordinate_frame_continuous(self, env_ids):
        """持续可视化坐标系轴和pitch向量"""
        # 只在display=True时才进行可视化
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
            
        # 清除之前的线条
        self.gym.clear_lines(self.viewer)
        
        # 为每个环境创建坐标系轴
        for i, env_id in enumerate(env_ids):
            # 1. 显示初始计算的坐标系（如果存在）
            if self._init_coordinate_frames is not None:
                init_pos = self._init_coordinate_frames[env_id, :3].cpu().numpy()
                init_rot_matrix = self._init_coordinate_frames[env_id, 3:12].reshape(3, 3).cpu().numpy()
                
                # 显示初始坐标系（橙色系）
                # self._draw_single_coordinate_frame(env_id, init_pos, init_rot_matrix, is_initial=True)
            
            # 2. 显示当前leg的实际坐标系
            leg_pos = self._fparts_states["leg"][env_id, :3].cpu().numpy()
            # 使用与compute_reward中相同的旋转信息
            leg_quat = self.states["leg_rot"][env_id].cpu().numpy()
            
            # 将四元数从xyzw顺序转换为wxyz顺序（quaternion_to_matrix期望wxyz顺序）
            leg_quat_wxyz = np.array([leg_quat[3], leg_quat[0], leg_quat[1], leg_quat[2]])
            
            # 将四元数转换为旋转矩阵
            leg_rot_tensor = torch.tensor(leg_quat_wxyz, device=self.sim_device).unsqueeze(0)
            leg_rot = torch_jit_utils.quaternion_to_matrix(leg_rot_tensor).squeeze(0).cpu().numpy()
            
            # 显示当前坐标系（红色系）
            self._draw_single_coordinate_frame(env_id, leg_pos, leg_rot, is_initial=False)
            
            # 3. 显示pitch向量（如果已计算）
            if hasattr(self, '_leg_pitch_vec') and hasattr(self, '_eef_pitch_vec'):
                leg_pitch_vec = self._leg_pitch_vec[env_id].cpu().numpy()
                eef_pitch_vec = self._eef_pitch_vec[env_id].cpu().numpy()
                # 使用绝对位置（加上base位置）
                leg_pos_vec = self.states["leg_pos"][env_id].cpu().numpy() + self._base_state[env_id, :3].cpu().numpy()
                eef_pos_vec = self.states["eef_pos"][env_id].cpu().numpy() + self._base_state[env_id, :3].cpu().numpy()
                
                # 显示pitch向量
                # self._draw_pitch_vectors(env_id, leg_pos_vec, eef_pos_vec, leg_pitch_vec, eef_pitch_vec)

    def _draw_single_coordinate_frame(self, env_id, position, rotation, is_initial=False):
        """绘制单个坐标系"""
        # 根据坐标系类型设置不同的参数
        if is_initial:
            axis_length = 0.15  # 初始坐标系更大
            colors = np.array([
                [1.0, 0.5, 0.0],  # 橙色 - X轴
                [0.5, 1.0, 0.0],  # 黄绿色 - Y轴
                [0.0, 0.5, 1.0],  # 青色 - Z轴
            ], dtype=np.float32)
        else:
            axis_length = 0.12  # 当前坐标系较大
            colors = np.array([
                [1.0, 0.0, 0.0],  # 红色 - X轴
                [0.0, 1.0, 0.0],  # 绿色 - Y轴
                [0.0, 0.0, 1.0],  # 蓝色 - Z轴
            ], dtype=np.float32)
        
        # 计算三个轴的方向向量
        x_axis = rotation[:, 0] * axis_length
        y_axis = rotation[:, 1] * axis_length
        z_axis = rotation[:, 2] * axis_length
        
        # 创建轴的可视化
        # 准备顶点数据：每个轴需要2个点（起点和终点）
        vertices = np.array([
            [position[0], position[1], position[2], position[0] + x_axis[0], position[1] + x_axis[1], position[2] + x_axis[2]],  # X轴
            [position[0], position[1], position[2], position[0] + y_axis[0], position[1] + y_axis[1], position[2] + y_axis[2]],  # Y轴
            [position[0], position[1], position[2], position[0] + z_axis[0], position[1] + z_axis[1], position[2] + z_axis[2]],  # Z轴
        ], dtype=np.float32)
        
        # 添加线条到viewer
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            3,  # 3条线
            vertices,
            colors
        )

    def _draw_pitch_vectors(self, env_id, leg_pos, eef_pos, leg_pitch_vec, eef_pitch_vec):
        """绘制leg和eef的pitch向量"""
        # 向量长度
        vec_length = 0.2
        
        # 计算向量的终点
        leg_vec_end = leg_pos + leg_pitch_vec * vec_length
        eef_vec_end = eef_pos + eef_pitch_vec * vec_length
        
        # 准备顶点数据：每个向量需要2个点（起点和终点）
        vertices = np.array([
            [leg_pos[0], leg_pos[1], leg_pos[2], leg_vec_end[0], leg_vec_end[1], leg_vec_end[2]],  # leg pitch向量
            [eef_pos[0], eef_pos[1], eef_pos[2], eef_vec_end[0], eef_vec_end[1], eef_vec_end[2]],  # eef pitch向量
        ], dtype=np.float32)
        
        # 设置颜色：leg向量为紫色，eef向量为青色
        colors = np.array([
            [0.8, 0.0, 0.8],  # 紫色 - leg pitch向量
            [0.0, 0.8, 0.8],  # 青色 - eef pitch向量
        ], dtype=np.float32)
        
        # 添加线条到viewer
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            2,  # 2条线
            vertices,
            colors
        )

    def compute_reward(self, actions):
        # 计算leg和eef的pitch向量用于可视化
        leg_rot = self.states["leg_rot"]
        eef_rot = self.states["eef_quat"]
        
        # 将四元数从xyzw顺序转换为wxyz顺序（quaternion_to_matrix期望wxyz顺序）
        leg_rot_wxyz = torch.cat([leg_rot[:, -1:], leg_rot[:, :-1]], dim=-1)  # xyzw -> wxyz
        eef_rot_wxyz = torch.cat([eef_rot[:, -1:], eef_rot[:, :-1]], dim=-1)  # xyzw -> wxyz
        
        # 将四元数转换为旋转矩阵
        leg_rot_matrix = torch_jit_utils.quaternion_to_matrix(leg_rot_wxyz)  # (N, 3, 3)
        eef_rot_matrix = torch_jit_utils.quaternion_to_matrix(eef_rot_wxyz)  # (N, 3, 3)
        
        # 获取两个向量的Z轴方向（pitch向量）
        leg_blue_z_vec = leg_rot_matrix[:, :, 2]  # (N, 3) - leg的Z轴方向
        eef_blue_z_vec = eef_rot_matrix[:, :, 2]  # (N, 3) - eef的Z轴方向
        
        # 获取leg和eef的位置用于可视化
        leg_pos = self.states["leg_pos"]
        eef_pos = self.states["eef_pos"]
        
        # 存储向量信息用于可视化
        self._leg_pitch_vec = leg_blue_z_vec
        self._eef_pitch_vec = eef_blue_z_vec
        self._leg_pos = leg_pos
        self._eef_pos = eef_pos
        
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
        ) = compute_reach_and_grasp_single_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            init_pitch_difference=self._init_pitch_difference,
            action=actions,
            states=self.states,
            max_episode_length=self.max_episode_length,
            angle_reward=self._angle_reward,
            success_reward=self._success_reward,
            qd_penalty=self._qd_penalty,
            action_penalty=self._action_penalty,
            ori_error_threshold=self._ori_error_threshold,
        )

@torch.jit.script
def compute_reach_and_grasp_single_reward(
    reset_buf,
    progress_buf,
    action,
    init_pitch_difference,
    states: dict[str, torch.Tensor],
    max_episode_length: int,
    angle_reward: float,
    success_reward: float,
    qd_penalty: float,
    action_penalty: float,
    ori_error_threshold: float,
):
    #下面的东西全都是张量,第一维都是env_nums
    #桌角的位置旋转角度
    leg_rot = states["leg_rot"]
    #夹爪的旋转角度
    eef_rot = states["eef_quat"]
    #夹爪之间的间距
    gripper_width = states["gripper_width"]
    if gripper_width.dim() > 1:
        gripper_width = gripper_width.squeeze(-1)
    #判断leg是否在夹爪手中
    grasp_leg_mask = gripper_width > 0.01

    failed = gripper_width < 0.01

    # 计算leg和eef的pitch向量（Y轴方向）
    # 将四元数从xyzw顺序转换为wxyz顺序（quaternion_to_matrix期望wxyz顺序）
    leg_rot_wxyz = torch.cat([leg_rot[:, -1:], leg_rot[:, :-1]], dim=-1)  # xyzw -> wxyz
    eef_rot_wxyz = torch.cat([eef_rot[:, -1:], eef_rot[:, :-1]], dim=-1)  # xyzw -> wxyz
    
    # 将四元数转换为旋转矩阵
    leg_rot_matrix = torch_jit_utils.quaternion_to_matrix(leg_rot_wxyz)  # (N, 3, 3)
    eef_rot_matrix = torch_jit_utils.quaternion_to_matrix(eef_rot_wxyz)  # (N, 3, 3)
    
    # 这个
    leg_blue_z_vec = leg_rot_matrix[:, :, 2]  # (N, 3) - leg的Y轴方向
    eef_blue_z_vec = eef_rot_matrix[:, :, 2]  # (N, 3) - eef的Y轴方向

    # 计算两个pitch向量之间的角度差
    # 使用点积计算夹角：cos(angle) = dot(a, b) / (|a| * |b|)
    dot_product = torch.sum(leg_blue_z_vec * eef_blue_z_vec, dim=-1)  # (N,)
    leg_norm = torch.norm(leg_blue_z_vec, dim=-1)  # (N,)
    eef_norm = torch.norm(eef_blue_z_vec, dim=-1)  # (N,)
    
    # 计算夹角（弧度）
    cos_angle = dot_product / (leg_norm * eef_norm) 
    angle_diff_rad = torch.acos(cos_angle)  # (N,)
    
    # 转换为角度
    pitch_diff = abs(90 - (angle_diff_rad * 180.0 / 3.1415926))  # (N,) - 单位是度

    succeeded = (pitch_diff < ori_error_threshold)
    succeeded = succeeded * grasp_leg_mask

    # 归一化角度差: 0 表示初始差，1 表示完全对齐
    # 确保所有张量都是一维的
    if init_pitch_difference.dim() > 1:
        init_pitch_difference = init_pitch_difference.view(-1)
    if pitch_diff.dim() > 1:
        pitch_diff = pitch_diff.view(-1)
    
    angle_progress = 1.0 - (pitch_diff / init_pitch_difference)  # 越接近 1 越好
    angle_progress = torch.clamp(angle_progress, min=-1.0, max=1.0)
    # qd penalty
    qd_norm = torch.linalg.norm(states["dq"], dim=-1)  # (num_envs,)

    # action penalty
    action_norm = torch.linalg.norm(action, dim=-1)

    rewards = (success_reward * succeeded 
        + angle_reward * angle_progress  
        - qd_penalty * qd_norm
        - action_penalty * action_norm)# (N,)
    # 如果达到最大步数或成功，则重置环境
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | succeeded | failed,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    return rewards, reset_buf, succeeded
