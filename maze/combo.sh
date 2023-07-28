maze=./config/maze2_simple_moredata.json
horizon=200
dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0726-114734_smd_stable/model"
# dynamics_path=None
data_file=./dataset/maze2_smd_stable.dat
mdp_ckpt_dir=./checkpoint/maze2_smd_stable
ckpt_path=${mdp_ckpt_dir}/outputpolicy
# rollout_epochs=1000
rollout_epochs=20
goal_mul=1

python scripts/run_combostyle.py --maze_config_file ${maze} \
                                 --horizon ${horizon} \
                                 --load-dynamics-path ${dynamics_path} \
                                 --data_file ${data_file} \
                                 --mdp_ckpt_dir ${mdp_ckpt_dir} \
                                 --ckpt_path ${ckpt_path} \
                                 --rollout_epochs ${rollout_epochs} \
                                 --goal_mul ${goal_mul} \
                                 --test_rollout \
                                 --init_state \
                                 --debug \
                                 --behavior_type mlp