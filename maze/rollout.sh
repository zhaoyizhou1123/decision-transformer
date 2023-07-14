support=10
ckpt_name=test_reward
ckpt_root=./checkpoint
ckpt_dir=${ckpt_root}/${ckpt_name}
epoch=50
horizon=400

# python scripts/learn_mdp_main.py --ckpt_root ${ckpt_root} --ckpt_name ${ckpt_name} --n_support ${support}
python scripts/rollout.py --ckpt_dir ${ckpt_dir} --rollout_epochs $epoch --horizon $horizon --debug