<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1 style="font-size:1.76rem">
        Value Flows
      </h1>
    </summary>
  </ul>
</div>

</div>

## Overview

Value Flows is an RL algorithm using flow-matching methods to model the return distribution.

This repository contains code for running the Value Flows algorithm and four baselines: FBRAC, C51, IQN, and CODAC. For other baselines, the implementations can be found in the official [FQL](https://github.com/seohongpark/fql) repository.

## Installation

1. Create an Anaconda environment: `conda create -n value-flows python=3.10.13 -y`
2. Activate the environment: `conda activate value-flows`
3. Install the dependencies:
   ```
   conda install -c conda-forge glew -y
   conda install -c conda-forge mesalib -y
   pip install -r requirements.txt
   ```
3. Setup `MuJoCo 2.1.0` for D4RL environments
   ```
   mkdir ~/.mujoco
   cd ~/.mujoco
   wget https://roboti.us/file/mjkey.txt
   wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
   tar -xvzf mujoco210-linux-x86_64.tar.gz
   rm mujoco210-linux-x86_64.tar.gz
   ```
4. Export environment variables
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/cz8792/.mujoco/mujoco210/bin
   export PYTHONPATH=path_to_value_flows_dir
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   ```

## Running experiments

```

# cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fdrl.py --agent.discount=0.995 --agent.alpha_critic_td_vf=1 --agent.critic_loss_type=q-learning --agent.next_action_extraction=sfbc --agent.policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=mean --seed=10

# C51
python main.py --env_name=cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/c51.py --agent.discount=0.995 --agent.num_atoms=101 --agent.q_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=True --seed=10

# IQN
python main.py --env_name=cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/iqn.py --agent.discount=0.995 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=False --seed=10

# CODAC
python main.py --env_name=cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/codac.py --agent.discount=0.995 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=300 --agent.alpha_penalty=0.1 --agent.normalize_q_loss=False --seed=10

# puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=2 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3  --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=mean --seed=10

# C51
python main.py --env_name=puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/c51.py --agent.discount=0.995 --agent.num_atoms=101 --agent.q_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=True --seed=10

# IQN
python main.py --env_name=puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/iqn.py --agent.discount=0.995 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=False --seed=10

# CODAC
python main.py --env_name=puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/codac.py --agent.discount=0.995 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=1000 --agent.alpha_penalty=0.1 --agent.normalize_q_loss=False --seed=10

# scene-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3  --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=mean --seed=10

# C51
python main.py --env_name=scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=True --seed=10

# IQN
python main.py --env_name=scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=False --seed=10

# CODAC
python main.py --env_name=scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=100 --agent.alpha_penalty=0.1 --agent.normalize_q_loss=False --seed=10


# puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=100 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10

# C51
python main.py --env_name=puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=101 --agent.q_agg=min --agent.actor_loss=sfbc --agent.normalize_q_loss=True --seed=10

# IQN
python main.py --env_name=puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=False --seed=10

# CODAC
python main.py --env_name=puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=1000 --agent.alpha_penalty=0.1 --agent.normalize_q_loss=False --seed=10

# cube-triple-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=cube-triple-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fdrl.py --agent.discount=0.995 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.03 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=mean --seed=10

# FQL
python main.py --env_name=cube-triple-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fql.py --agent.discount=0.995 --agent.alpha=300 --agent.q_agg=mean --seed=10

# C51
python main.py --env_name=cube-triple-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/c51.py --agent.discount=0.995 --agent.num_atoms=51 --agent.q_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=True --seed=10

# IQN
python main.py --env_name=cube-triple-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/iqn.py --agent.discount=0.995 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=sfbc --agent.normalize_q_loss=False --seed=10

# CODAC
python main.py --env_name=cube-triple-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/codac.py --agent.discount=0.995 --agent.num_cosines=64 --agent.kappa=0.95 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=100 --agent.alpha_penalty=0.1 --agent.normalize_q_loss=False --seed=10

# visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.03 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# FQL
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fql.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IQL
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.alpha=1.0 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# ReBRAC
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/rebrac.py --agent.discount=0.99 --agent.alpha_actor=0.003 --agent.alpha_critic=0.01 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IFQL
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/ifql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.num_samples=32 --agent.encoder=impala_small --seed=10

# FBRAC
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fbrac.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.normalize_q_loss=False --agent.encoder=impala_small --seed=10

# IQN
python main.py --env_name=visual-antmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.encoder=impala_small --seed=10

# visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0  --p_aug=0.5 --frame_stack=3 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.03 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# FQL
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fql.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IQL
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.alpha=1.0 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# ReBRAC
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/rebrac.py --agent.discount=0.99 --agent.alpha_actor=0.003 --agent.alpha_critic=0.01 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IFQL
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/ifql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.num_samples=32 --agent.encoder=impala_small --seed=10

# FBRAC
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fbrac.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.normalize_q_loss=False --agent.encoder=impala_small --seed=10

# IQN
python main.py --env_name=visual-antmaze-teleport-navigate-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.encoder=impala_small --seed=10

# visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fdrl.py --agent.discount=0.995 --agent.alpha_critic_td_vf=1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# FQL
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fql.py --agent.discount=0.995 --agent.alpha=100 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IQL
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0  --p_aug=0.5 --frame_stack=3 --agent=agents/iql.py --agent.discount=0.995 --agent.expectile=0.9 --agent.alpha=0.3 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# ReBRAC
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/rebrac.py --agent.discount=0.995 --agent.alpha_actor=0.1 --agent.alpha_critic=0 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IFQL
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/ifql.py --agent.discount=0.995 --agent.expectile=0.9 --agent.num_samples=32 --agent.encoder=impala_small --seed=10

# FBRAC
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fbrac.py --agent.discount=0.995 --agent.alpha=100 --agent.q_agg=mean --agent.normalize_q_loss=False --agent.encoder=impala_small --seed=10

# IQN
python main.py --env_name=visual-cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.encoder=impala_small --seed=10

# visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# FQL
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fql.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IQL
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0  --p_aug=0.5 --frame_stack=3 --agent=agents/iql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.alpha=10.0 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# ReBRAC
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/rebrac.py --agent.discount=0.99 --agent.alpha_actor=0.1 --agent.alpha_critic=0.01 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IFQL
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/ifql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.num_samples=32 --agent.encoder=impala_small --seed=10

# FBRAC
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fbrac.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.normalize_q_loss=False --agent.encoder=impala_small --seed=10

# IQN
python main.py --env_name=visual-scene-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.95 --agent.encoder=impala_small --seed=10

# visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0
# FDRL
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --agent=agents/fdrl.py --p_aug=0.5 --frame_stack=3 --agent.discount=0.99 --agent.alpha_critic_td_vf=3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# FQL
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fql.py --agent.discount=0.99 --agent.alpha=100 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IQL
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0  --p_aug=0.5 --frame_stack=3 --agent=agents/iql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.alpha=10.0 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# ReBRAC
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/rebrac.py --agent.discount=0.99 --agent.alpha_actor=0.3 --agent.alpha_critic=0.01 --agent.q_agg=mean --agent.encoder=impala_small --seed=10

# IFQL
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/ifql.py --agent.discount=0.99 --agent.expectile=0.9 --agent.num_samples=32 --agent.encoder=impala_small --seed=10

# FBRAC
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/fbrac.py --agent.discount=0.99 --agent.alpha=300 --agent.q_agg=mean --agent.normalize_q_loss=False --agent.encoder=impala_small --seed=10

# IQN
python main.py --env_name=visual-puzzle-3x3-play-singletask-{task1, task2, task3, task4, task5}-v0 --p_aug=0.5 --frame_stack=3 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.encoder=impala_small --seed=10

# pen-{human, cloned, expert}-v1
# FDRL
python main.py --env_name=pen-human-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=min --seed=10
python main.py --env_name=pen-cloned-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10
python main.py --env_name=pen-expert-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10

# IQN
python main.py --env_name=pen-human-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=pen-cloned-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=pen-expert-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10

# C51
python main.py --env_name=pen-human-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=pen-cloned-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=pen-expert-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10

# CODAC
python main.py --env_name=pen-human-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=pen-cloned-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.1 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=pen-expert-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10

# door-{human, cloned, expert}-v1
# FDRL
python main.py --env_name=door-human-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10
python main.py --env_name=door-cloned-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=min --seed=10
python main.py --env_name=door-expert-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=min --seed=10

# IQN
python main.py --env_name=door-human-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=door-cloned-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=door-expert-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.normalize_q_loss=True --seed=10

# C51
python main.py --env_name=door-human-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=101 --agent.q_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=door-cloned-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=door-expert-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=mean --agent.normalize_q_loss=True --seed=10

# CODAC
python main.py --env_name=door-human-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=door-cloned-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=30000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=door-expert-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10

# hammer-{human, cloned, expert}-v1
# FDRL
python main.py --env_name=hammer-human-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10
python main.py --env_name=hammer-cloned-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10
python main.py --env_name=hammer-expert-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=min --seed=10

# IQN
python main.py --env_name=hammer-human-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.7 --agent.quantile_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=hammer-cloned-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.7 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=hammer-expert-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.7 --agent.quantile_agg=mean --agent.normalize_q_loss=True --seed=10

# C51
python main.py --env_name=hammer-human-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=hammer-cloned-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=hammer-expert-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10

# CODAC
python main.py --env_name=hammer-human-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=30000 --agent.penalty=0.1 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=hammer-cloned-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=hammer-expert-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.8 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10

# relocate-{human, cloned, expert}-v1
# FDRL
python main.py --env_name=relocate-human-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.1 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=min --seed=10
python main.py --env_name=relocate-cloned-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=mean --agent.q_agg=min --seed=10
python main.py --env_name=relocate-expert-v1 --agent=agents/fdrl.py --agent.discount=0.99 --agent.alpha_critic_td_vf=0.3 --agent.critic_loss_type=q-learning --next_action_extraction=sfbc --policy_extraction=sfbc --agent.ensemble_weight_type=target_ret_std_jac_est --agent.ensemble_weight_temp=0.3 --agent.clip_flow_actions=True --agent.value_layer_norm=True --agent.actor_layer_norm=True --agent.ret_agg=min --agent.q_agg=min --seed=10

# IQN
python main.py --env_name=relocate-human-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=relocate-cloned-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.normalize_q_loss=True --seed=10
python main.py --env_name=relocate-expert-v1 --agent=agents/iqn.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=min --agent.normalize_q_loss=True --seed=10

# C51
python main.py --env_name=relocate-human-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=101 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=relocate-cloned-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=51 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10
python main.py --env_name=relocate-expert-v1 --agent=agents/c51.py --agent.discount=0.99 --agent.num_atoms=101 --agent.q_agg=min --agent.normalize_q_loss=True --seed=10

# CODAC
python main.py --env_name=relocate-human-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=30000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=relocate-cloned-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=30000 --agent.penalty=0.01 --agent.normalize_q_loss=False --seed=10
python main.py --env_name=relocate-expert-v1 --agent=agents/codac.py --agent.discount=0.99 --agent.num_cosines=64 --agent.kappa=0.9 --agent.quantile_agg=mean --agent.actor_loss=ddpgbc --agent.alpha=10000 --agent.penalty=0.1 --agent.normalize_q_loss=False --seed=10
    
</details>
