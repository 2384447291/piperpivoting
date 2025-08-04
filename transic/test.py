# hydra：配置管理库，用于从 .yaml 文件动态注入配置
import hydra
#辅助hydra库的一些包
from omegaconf import DictConfig, OmegaConf
import transic
import numpy as np


# 定义一个函数 preprocess_train_config，用于在通用位置给 RL-games 的训练配置 
# (config_dict) 添加一些公共字段，例如设备、PBT 等。这样可以避免在每个任务的 YAML 文件中重复相同的配置。
def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """
    #从 config_dict中提取出 RL-training 要用的 params.config 部分，存到 train_cfg
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None
    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    # Using rl_device: cuda:0
    # Using sim_device: cuda:0
    # {'name': 'InsertFull', 
    # 'full_experiment_name': None, '
    # env_name': 'rlgpu', 
    # 'multi_gpu': False, 
    # 'ppo': True, 
    # 'mixed_precision': False, 
    # 'normalize_input': True, 
    # 'normalize_input_excluded_keys': [], 
    # 'normalize_value': True, 'value_bootstrap': True, 'num_actors': 512, 
    # 'reward_shaper': {'scale_value': 1.0}, 
    # 'normalize_advantage': True, 'gamma': 0.99, 'tau': 0.95, 'learning_rate': 0.0005, '
    # lr_schedule': 'adaptive', 'schedule_type': 'standard', 'kl_threshold': 0.008, 
    # 'score_to_win': 10000, 'max_epochs': 9999999999999, 'save_best_after': 200, 
    # 'save_frequency': 100, 'print_stats': True, 'grad_norm': 1.0, 'entropy_coef': 0.0, 
    # 'truncate_grads': True, 'e_clip': 0.2, 'horizon_length': 32, 'minibatch_size': 4096, 
    # 'mini_epochs': 5, 'critic_coef': 4, 'clip_value': True, 'seq_len': 4, 'bounds_loss_coef': 0.0001, 
    # 'device': 'cuda:0', 'population_based_training': False, 'pbt_idx': None}


    # 尝试读取多层感知机（MLP）网络配置中的 model_size_multiplier。
    # 如果该倍率不为 1，则遍历 units（每层神经元数），将其乘以该倍率， 实现网络宽度可调。
    # 捕获 KeyError 是为了在配置中缺少该字段时跳过此段逻辑。
    # 返回修改后的完整配置字典。
    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"][
            "model_size_multiplier"
        ]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="main/cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import os
    from datetime import datetime

    import isaacgym
    from hydra.utils import to_absolute_path

    import transic_envs
    from transic.utils.reformat import omegaconf_to_dict, print_dict
    from transic.utils.utils import set_np_formatting, set_seed

    from transic.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        MultiObserver,
        ComplexObsRLGPUEnv,
    )
    from transic.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from transic.rl.runner import Runner
    from transic.rl.network_builder import DictObsBuilder
    from transic.rl.models import ModelA2CContinuousLogStd
    from rl_games.algos_torch.model_builder import register_network, register_model
    from transic.utils.wandb_utils import WandbVideoCaptureWrapper
    from isaacgym import gymapi  # 修复导入
    
    #在 RL-Games 框架中注册自定义的模型和网络构建器，以便在配置中通过名称引用。
    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_network("dict_obs_actor_critic", DictObsBuilder)

    # ensure checkpoints can be specified as relative paths
    # 如果配置中指定了检查点路径，则将其转换为绝对路径, 方便从本地加载模型。
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    # 创建一个函数 create_isaacgym_env，用于创建 Isaac Gym 环境。
    def create_isaacgym_env():
        kwargs = dict(
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            multi_gpu=cfg.multi_gpu,
            cfg=cfg.task,
            display=cfg.display,
            record=cfg.capture_video,
            has_headless_arg=False,
        )
        if not cfg.headless:
            assert (
                "pcd" not in cfg.task_name.lower()
            ), "TODO: add GUI support for PCD tasks"
        if "pcd" not in cfg.task_name.lower():
            kwargs["headless"] = cfg.headless
            kwargs["has_headless_arg"] = True
        envs = transic_envs.make(**kwargs)
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = WandbVideoCaptureWrapper(
                envs,
                n_parallel_recorders=cfg.n_parallel_recorders,
                n_successful_videos_to_record=cfg.n_successful_videos_to_record,
            )
        return envs

    # 直接调用create_isaacgym_env()来显示仿真环境
    print("开始创建Piper环境...")
    envs = create_isaacgym_env()
    print(f"环境创建成功!")
    print(f"环境数量: {envs.num_envs}")
    print(f"动作空间维度: {envs.num_actions}")
    print(f"设备: {envs.device}")
    
    # 使用环境的标准接口进行可视化测试
    print("\n=== 使用环境标准接口进行可视化测试 ===")
    
    # 1. 重置环境获取初始观察
    print("1. 重置环境...")
    obs = envs.reset()
    print(f"观察字典的键: {list(obs.keys())}")
    print(f"观察字典内容: {obs}")
    
    # 检查观察结构
    if 'obs' in obs:
        print(f"初始观察形状: {obs['obs'].shape}")
    elif 'proprioception' in obs:
        print(f"初始本体感受观察形状: {obs['proprioception'].shape}")
    else:
        print(f"观察结构: {obs}")
    
    print(f"观察空间: {envs.observation_space}")
    print(f"动作空间: {envs.action_space}")
    
    # 2. 显示环境状态信息
    print("\n2. 环境状态信息:")
    print(f"可用状态键: {list(envs.states.keys())}")
    
    
    # 3. 设置可视化
    print("\n3. 设置可视化界面...")
    
    # 检查是否有GUI支持
    if hasattr(envs, 'viewer') and envs.viewer is not None:
        print("✓ Isaac Gym GUI 已启用")
    elif cfg.display:
        print("⚠ Isaac Gym GUI 未启用，但display=true已设置")
        print("💡 提示：确保graphics_device_id设置正确，并且系统支持GPU渲染")
    else:
        print("⚠ Isaac Gym GUI 未启用，请设置 display=true")
    
    # 4. 开始键盘控制
    print("\n4. 开始键盘控制...")
    import torch
    import time
    
    # 键盘控制说明 - 适配OSC控制器
    print("🎮 键盘控制说明 (OSC控制器):")
    print("  W/S - 控制末端执行器前后移动 (X轴位置增量)")
    print("  A/D - 控制末端执行器左右移动 (Y轴位置增量)")
    print("  Q/E - 控制末端执行器上下移动 (Z轴位置增量)")
    print("  I/K - 控制末端执行器绕X轴旋转 (俯仰)")
    print("  J/L - 控制末端执行器绕Y轴旋转 (偏航)")
    print("  U/O - 控制末端执行器绕Z轴旋转 (翻滚)")
    print("  Space - 夹爪开关切换")
    print("  R - 重置环境")
    print("  ESC - 退出程序")
    print("\n💡 提示：按任意键开始控制，按ESC退出")
    
    # 控制变量 - 适配7维OSC动作
    current_action = torch.zeros(envs.num_envs, envs.num_actions, device=envs.device)
    action_scale = 0.8  # 位置增量幅度 (米)
    rotation_scale = 0.8# 旋转增量幅度 (弧度)
    
    # 夹爪状态
    gripper_open = True  # True=打开, False=关闭
    
    # 显示episode信息
    print(f"📊 Episode信息:")
    print(f"  - 最大步数: {envs.max_episode_length}")
    print(f"  - 当前步数: 0")
    print(f"  - 环境数量: {envs.num_envs}")
    print(f"  - 动作维度: {envs.num_actions} (OSC控制器)")
    print(f"💡 提示: 每{envs.max_episode_length}步环境会自动重置，按R键可手动重置")
    
    # 初始化循环变量
    step = 0
    episode_step = 0  # 当前episode的步数
    
    # 检查是否有viewer，如果有则使用Isaac Gym的标准渲染循环
    if hasattr(envs, 'viewer') and envs.viewer is not None:
        print("🎮 使用Isaac Gym标准渲染循环...")
        print("💡 提示：您现在可以自由操作 Isaac Gym 界面，无需按键也能看到仿真运行")
        print("💡 提示：键盘控制是可选的，不会阻塞界面操作")
        print("💡 提示：按 ESC 键或关闭窗口退出")
        
        # 初始化时间变量
        if not hasattr(envs, 'last_frame_time'):
            envs.last_frame_time = time.time()
        
        # 确保viewer同步已启用
        if not hasattr(envs, 'enable_viewer_sync'):
            envs.enable_viewer_sync = True
        
        print(f"Viewer状态: {envs.viewer}")
        print(f"Viewer同步: {envs.enable_viewer_sync}")
        print(f"Sim状态: {envs.sim}")
        
        while not envs.gym.query_viewer_has_closed(envs.viewer):
            # 执行动作（即使没有动作也要执行step来推进仿真）
            obs, rewards, resets, info = envs.step(current_action)
            
            # 更新episode步数
            episode_step += 1
            
            
            # 更新图形 - 这部分必须每次都执行
            if envs.device != "cpu":
                envs.gym.fetch_results(envs.sim, True)
            
            # 步进图形 - 这部分必须每次都执行
            if envs.enable_viewer_sync:
                envs.gym.step_graphics(envs.sim)
                envs.gym.draw_viewer(envs.viewer, envs.sim, True)
                
                # 等待dt时间流逝，同步物理仿真与渲染速率
                envs.gym.sync_frame_time(envs.sim)
                
                # 控制渲染帧率
                now = time.time()
                delta = now - envs.last_frame_time
                if hasattr(envs, 'render_fps') and envs.render_fps < 0:
                    render_dt = envs.dt * envs.control_freq_inv
                elif hasattr(envs, 'render_fps'):
                    render_dt = 1.0 / envs.render_fps
                else:
                    render_dt = 0.016  # 默认60fps
                
                if delta < render_dt:
                    time.sleep(render_dt - delta)
                
                envs.last_frame_time = time.time()
            else:
                envs.gym.poll_viewer_events(envs.viewer)
        
        print("Viewer窗口已关闭")
        
    else:
        # 没有viewer时的简化循环
        print("🎮 使用简化循环（无GUI）...")
        print("💡 提示：无GUI模式，仿真将自动运行")
        print("💡 提示：按 Ctrl+C 退出程序")
        
        try:
            while True:
                # 执行动作（无GUI模式下自动运行）
                obs, rewards, resets, info = envs.step(current_action)
                
                # 更新episode步数
                episode_step += 1
                
                # 每50步打印一次状态
                if step % 50 == 0:
                    eef_pos = envs.states['eef_pos'][0].detach().cpu().numpy()
                    eef_quat = envs.states['eef_quat'][0].detach().cpu().numpy()
                    dof_pos = envs.states['q'][0].detach().cpu().numpy()
                    print(f'Step {step}, Episode步数: {episode_step}/{envs.max_episode_length}')
                    print(f'Step {step}, 末端执行器位置: {eef_pos}')
                    print(f'Step {step}, 末端执行器姿态: {eef_quat}')
                    print(f'Step {step}, 关节角度: {dof_pos}')
                    print(f'Step {step}, 当前动作: {current_action[0].detach().cpu().numpy()}')
                    print(f'Step {step}, 夹爪状态: {"打开" if gripper_open else "关闭"}')
                    print(f'Step {step}, 奖励: {rewards[0].detach().cpu().numpy()}')
                
                # 检查是否需要重置
                if resets.any():
                    reset_env_ids = torch.where(resets)[0].detach().cpu().numpy()
                    print(f"🔄 Episode结束，环境自动重置! 重置的环境ID: {reset_env_ids}")
                    print(f"📊 Episode统计: 总步数={step}, Episode步数={episode_step}")
                    episode_step = 0  # 重置episode步数计数器
                
                step += 1
                time.sleep(0.01)  # 控制循环频率
                
        except KeyboardInterrupt:
            print("\n用户按下了 Ctrl+C，退出程序")
    
    print("\n=== 键盘控制结束! ===")


if __name__ == "__main__":
    import os
    os.environ["HTTP_PROXY"] = "http://192.168.99.32:2333"
    os.environ["HTTPS_PROXY"] = "http://192.168.99.32:2333"
    launch_rlg_hydra()