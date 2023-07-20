dynamics_path="./log/pointmaze/combo/seed_1&timestamp_23-0719-231116/model"
# dynamics_path=None
data_file=./dataset/maze_1e6_nonstop_randomend.dat
mdp_ckpt_dir=./checkpoint/randomend_1e6

python scripts/run_combostyle.py --load-dynamics-path ${dynamics_path} --data_file ${data_file} --mdp_ckpt_dir ${mdp_ckpt_dir}