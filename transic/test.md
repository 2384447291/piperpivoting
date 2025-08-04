PiperPivoting
ReachAndGraspSingle


python3 main/rl/train.py task=PiperPivoting num_envs=2048 sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 wandb_activate=true wandb_entity=wendyhgnewman-southern-university-of-science-technology wandb_project=transcl display=false headless=true

python3 main/rl/train.py task=PiperPivoting num_envs=1 test=true checkpoint="./runs/PiperPivoting_08-02-03-10-39/nn/PiperPivoting.pth" display=true headless=true

python3 test.py task=PiperPivoting num_envs=1  sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 display=true headless=false