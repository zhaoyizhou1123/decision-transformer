maze=./config/maze2_simple_moredata.json
algo=stitch-cql
horizon=200
# dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0727-123603_smds_acc/model"
# dynamics_path=None
data_file=./dataset/maze2_smds_acc.dat
# ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
rollout_epochs=5000
# rollout_epochs=20
num_need_traj=100

behavior_epoch=50
goal_mul=1

num_diffusion_iters=10
epochs=50


for offline_ratio in 0.25 0.5 0.75 1
    do
    for dyn_seed in 0 1 2 3
    do
        # mdp_ckpt_dir=./checkpoint/maze2-stitch-mlp_${dyn_seed} # No use for diffusion
        d_seed=maze2-stitch-mlp_${dyn_seed}
        dynamics_path="./log/pointmaze/combo/seed_${dyn_seed}&timestamp_23-0810_keep/model"

        final_ckpt_path=None
        rollout_ckpt_path=./checkpoint/maze2-stitch-mlp_${dyn_seed}

        python scripts/run_combostyle.py --maze_config_file ${maze} \
                                        --cql_seed ${dyn_seed} --seed ${dyn_seed} \
                                        --horizon ${horizon} \
                                        --final_ckpt_path ${final_ckpt_path} \
                                        --load-dynamics-path ${dynamics_path} \
                                        --data_file ${data_file} \
                                        --rollout_epochs ${rollout_epochs} \
                                        --goal_mul ${goal_mul} \
                                        --algo ${algo} \
                                        --offline_ratio ${offline_ratio} \
                                        --diffusion_seed ${d_seed} \
                                        --num_diffusion_iters ${num_diffusion_iters} \
                                        --behavior_epoch ${behavior_epoch} \
                                        --rollout_ckpt_path ${rollout_ckpt_path} \
                                        --num_need_traj ${num_need_traj} \
                                        --epochs ${epochs} \
                                        --log_to_wandb &
    done
    wait
done