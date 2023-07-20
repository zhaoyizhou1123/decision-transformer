dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0720-110808/model"
# dynamics_path=None
data_file=./dataset/maze_1e6_nonstop_randomend.dat
mdp_ckpt_dir=./checkpoint/randomend_1e6
ckpt_prefix=${mdp_ckpt_dir}/outputpolicy
# rollout_epochs=1000
rollout_epochs=10

python scripts/run_combostyle.py --load-dynamics-path ${dynamics_path} --data_file ${data_file} --mdp_ckpt_dir ${mdp_ckpt_dir} --ckpt_prefix ${ckpt_prefix} --rollout_epochs ${rollout_epochs} --test_rollout