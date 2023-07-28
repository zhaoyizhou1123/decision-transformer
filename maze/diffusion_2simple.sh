maze=./config/maze2_simple_moredata.json
horizon=200
dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0727-123603_smds_acc/model"
# dynamics_path=None
data_file=./dataset/maze2_smds_acc.dat
mdp_ckpt_dir=./checkpoint/maze2_smds_accdyn # No use for diffusion
# ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
rollout_epochs=5000
# rollout_epochs=20
num_need_traj=10

behavior_epoch=50
goal_mul=1
d_seed=maze2_smds_accdyn
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
                                 --num_diffusion_iters ${num_diffusion_iters} \
                                 --behavior_epoch ${behavior_epoch} \
                                 --rollout_ckpt_path ${rollout_ckpt_path} \
                                 --num_need_traj ${num_need_traj} \
                                 --debug