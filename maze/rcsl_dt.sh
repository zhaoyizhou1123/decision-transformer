algo=rcsl-dt
maze=./config/maze2_simple_moredata.json
horizon=200
# dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0727-123603_smds_acc/model"
# dynamics_path=None
data_file=./dataset/maze2_smds_acc.dat
# data_file=./checkpoint/maze2_smds_accdyn/rollout.dat
mdp_ckpt_dir=./checkpoint/maze2simpleexert_mlp # No use for diffusion
# behavior_type=mlp
# ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
# rollout_epochs=5000
# rollout_epochs=20
# num_need_traj=100

# behavior_epoch=10
# # goal_mul=1
# d_seed=maze2_smds_accdyn
# num_diffusion_iters=10

epochs=100
ctx=20
# lr=1e-3


final_ckpt_path=${mdp_ckpt_dir}
# rollout_ckpt_path=${mdp_ckpt_dir}

# for arch in 1
# do
#     python scripts/run_rcsl.py --algo ${algo} \
#                             --epochs ${epochs} \
#                                 --maze_config_file ${maze} \
#                                 --horizon ${horizon} \
#                                 --final_ckpt_path ${final_ckpt_path} \
#                                 --data_file ${data_file}
# done 


for dyn_seed in 0 1 2 3
do
    # mdp_ckpt_dir=./checkpoint/maze2-stitch-mlp/ratio${offline_ratio}/seed${dyn_seed} # No use for diffusion
    d_seed=maze2-stitch-mlp_${dyn_seed}
    dynamics_path="./log/pointmaze/combo/seed_${dyn_seed}&timestamp_23-0810_keep/model"

    final_ckpt_path=./checkpoint/maze2-stitch-mlp/ratio${offline_ratio}/seed0
    rollout_ckpt_path=./checkpoint/maze2-stitch-mlp_${dyn_seed}

    python scripts/run_rcsl.py --algo ${algo} \
                                --epochs ${epochs} \
                                --seed ${dyn_seed} \
                                --maze_config_file ${maze} \
                                --horizon ${horizon} \
                                --final_ckpt_path ${final_ckpt_path} \
                                --data_file ${data_file} \
                                --num_workers 2 \
                                --ctx ${ctx} &
done
wait
# for width in 128 256 512 1024 2048 4096
# do
#     arch="${width}-${width}"
#     for dyn_seed in 0 1 2 3
#     do
#         python scripts/run_rcsl.py --algo ${algo} \
#                                     --epochs ${epochs} \
#                                     --seed ${dyn_seed} \
#                                     --maze_config_file ${maze} \
#                                     --horizon ${horizon} \
#                                     --final_ckpt_path ${final_ckpt_path} \
#                                     --data_file ${data_file} \
#                                     --num_workers 2 \
#                                     --arch ${arch} &
#         sleep 10
#     done
#     wait
# done