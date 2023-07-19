support=10
ckpt_name=layer4_stateonly
ckpt_root=./checkpoint
ckpt_dir=${ckpt_root}/${ckpt_name}
policy_type=expert
train_type=default
epoch=10
horizon=400

# python scripts/learn_mdp_main.py --ckpt_root ${ckpt_root} --ckpt_name ${ckpt_name} --n_support ${support}
python scripts/rollout.py --ckpt_dir ${ckpt_dir} --rollout_epochs $epoch --horizon $horizon --policy_type $policy_type --train_type ${train_type} --log_to_wandb