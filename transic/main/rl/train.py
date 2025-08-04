# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# hydra：配置管理库，用于从 .yaml 文件动态注入配置
import hydra
#辅助hydra库的一些包
from omegaconf import DictConfig, OmegaConf
import transic


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


@hydra.main(version_base="1.1", config_name="config", config_path="../cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import os
    from datetime import datetime

    import isaacgym
    from hydra.utils import to_absolute_path

    # 如果在配置中开启了可视化（display=True），这里创建一个 OpenCV 窗口并显示一个 1×1 像素的黑图，
    # 以便后续仿真窗口能够正确初始化和渲染
    if cfg.display:
        import cv2
        import numpy as np

        cv2.imshow("dummy", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)

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

    # 在 RL-Games 中注册新的环境类型 "rlgpu"，指定向量化类型 RLGPU 和环境创建函数。
    # 在后面具体的runner的时候会调用这个函数来创建环境。
    #     vecenv_type	对应实现	主要特点
    #     RAY	RayVecEnv	基于 Ray 集群框架的并行环境；适合跨机器／跨进程大规模并行。 
    #     MULTIAGENT	MultiAgentVecEnv	专门针对多智能体场景的并行封装；每个环境内部可管理多 agent，通过同一进程并行。 
    #     CULE	CuleEnv	NVIDIA CULE（CUDA-accelerated Atari）环境，直接在 GPU 上并行运行 Atari  游戏。 
    #     IsaacRlgWrapper 	RlGamesGpuEnv （或 RlGamesVecEnvWrapper）	将 Isaac Gym/Isaac Lab 环境包装为 RL‑Games 可用的 GPU 向量环境；负责把观测和动作缓冲区从仿真转到同一 CUDA 设备并裁剪。 
    #     RLGPU	ComplexObsRLGPUEnv（或自定义 GPU wrapper）	在 Isaac Gym 恰当支持复杂观测（dict obs）的同时，实现高效 GPU 并行；通常与 transic_envs.make 联动使用。
    env_configurations.register(
        "rlgpu",
        { 
            "vecenv_type": "RLGPU",
            "env_creator": create_isaacgym_env,
        },
    )

    # 如果在训练配置里定义了“中心价值网络”(central_value_config)，
    # 则构造一个 obs_spec["states"]，用来告诉 RL-Games 框架如何拼接或命名状态输入。
    obs_spec = {}
    if "central_value_config" in cfg.rl_train.params.config:
        critic_net_cfg = cfg.rl_train.params.config.central_value_config.network
        obs_spec["states"] = {
            "names": list(critic_net_cfg.inputs.keys()),
            "concat": not critic_net_cfg.name == "complex_net",
            "space_name": "state_space",
        }

    # 注册 RLGPU 向量化环境，使用 ComplexObsRLGPUEnv 作为环境创建函数。
    vecenv.register(
        "RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name)
    )

    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if cfg.test:
        prefix = "dump_" if cfg.save_rollouts else "test_"
        experiment_dir = os.path.join(
            "runs",
            prefix
            + cfg.rl_train.params.config.name
            + "_{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
    else:
        experiment_dir = os.path.join(
            "runs",
            cfg.rl_train.params.config.name
            + "_{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "from_ckpt_epoch": cfg.from_ckpt_epoch,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
            "save_rollouts": {
                "save_rollouts": cfg.save_rollouts,
                "rollout_saving_fpath": os.path.join(experiment_dir, "rollouts.hdf5"),
                "save_successful_rollouts_only": cfg.save_successful_rollouts_only,
                "num_rollouts_to_save": cfg.num_rollouts_to_save,
                "min_episode_length": cfg.min_episode_length,
            },
        }
    )


if __name__ == "__main__":
    import os
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    launch_rlg_hydra()
