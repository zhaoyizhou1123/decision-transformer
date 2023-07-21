maze=./config/maze2.json
horizon=300
# dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0719-231116/model"
dynamics_path=None
data_file=./dataset/maze2.dat
mdp_ckpt_dir=./checkpoint/maze2
ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
rollout_epochs=1000
# rollout_epochs=10
goal_mul=1

python scripts/run_combostyle.py --maze_config_file ${maze} \
                                 --horizon ${horizon} \
                                 --load-dynamics-path ${dynamics_path} \
                                 --data_file ${data_file} \
                                 --mdp_ckpt_dir ${mdp_ckpt_dir} \
                                 --ckpt_prefix ${ckpt_prefix} \
                                 --rollout_epochs ${rollout_epochs} \
                                 --goal_mul ${goal_mul}