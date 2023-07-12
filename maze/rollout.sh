support=10
ckpt_name=support${support}
ckpt_root=./checkpoint
ckpt_dir=${ckpt_root}/${ckpt_name}

python scripts/learn_mdp_main.py --ckpt_root ${ckpt_root} --ckpt_name ${ckpt_name} --n_support ${support}
python scripts/rollout.py --ckpt_dir ${ckpt_dir}