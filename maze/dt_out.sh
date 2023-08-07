algo=rcsl-dt
maze=./config/maze2_simple_moredata.json
horizon=200
dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0727-123603_smds_acc/model"
# dynamics_path=None
data_file=./dataset/maze2_smds_acc.dat
# data_file=./checkpoint/maze2_smds_accdyn/rollout.dat
mdp_ckpt_dir=./checkpoint/maze2smds_rolloutonly_dt_policy # No use for diffusion
behavior_type=mlp
# ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
rollout_epochs=5000
# rollout_epochs=20
num_need_traj=100

behavior_epoch=10
goal_mul=1.3
d_seed=maze2_smds_accdyn
num_diffusion_iters=10

epochs=10
# ctx=5
lr=1e-3


final_ckpt_path=${mdp_ckpt_dir}
rollout_ckpt_path=${mdp_ckpt_dir}

for ctx in 20
do
    python scripts/run_rcsl.py --algo ${algo} \
                            --ctx ${ctx} \
                            --rate ${lr} \
                            --epochs ${epochs} \
                                --maze_config_file ${maze} \
                                --horizon ${horizon} \
                                --final_ckpt_path ${final_ckpt_path} \
                                --data_file ${data_file} \
                                --goal_mul ${goal_mul} \
                                --debug
done 