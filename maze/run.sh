maze=./config/maze2_simple_moredata.json
horizon=200
data_file=./dataset/maze2_smds_acc.dat
goal_mul=1.35

# behavior_type=mlp
mdp_ckpt_dir=./checkpoint/maze2smds_mlp_b_policy

final_ckpt_path=${mdp_ckpt_dir}

python scripts/run_rcsl.py  --maze_config_file ${maze} \
                            --horizon ${horizon} \
                            --data_file ${data_file} \
                            --final_ckpt_path ${final_ckpt_path} \
                            --goal_mul ${goal_mul} \
                            --debug