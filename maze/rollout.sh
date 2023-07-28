maze=./config/maze2_simple_moredata.json
horizon=200
dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0726-114734_smd_stable/model"
# dynamics_path=None
data_file=./dataset/maze2_smd_stable.dat
mdp_ckpt_dir=./checkpoint/maze2_smd_stable # No use for diffusion
# ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
rollout_epochs=20
# rollout_epochs=20
behavior_epoch=50
goal_mul=1
d_seed=maze2_smd_stable
num_diffusion_iters=10

final_ckpt_path=${mdp_ckpt_dir}
rollout_ckpt_path=${mdp_ckpt_dir}

python scripts/run_combostyle.py --maze_config_file ${maze} \
                                 --horizon ${horizon} \
                                 --final_ckpt_path ${final_ckpt_path} \
                                 --load-dynamics-path ${dynamics_path} \
                                 --data_file ${data_file} \
                                 --mdp_ckpt_dir ${mdp_ckpt_dir} \
                                 --rollout_epochs ${rollout_epochs} \
                                 --goal_mul ${goal_mul} \
                                 --diffusion_seed ${d_seed} \
                                 --behavior_epoch ${behavior_epoch} \
                                 --num_diffusion_iters ${num_diffusion_iters} \
                                 --load_diffusion \
                                 --rollout_ckpt_path ${rollout_ckpt_path} \
                                 --test_rollout --init_state \
                                 --debug