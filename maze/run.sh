epoch=50
algo=stitch

python scripts/run.py --algo $algo --epochs $epoch --mdp_epochs $epoch --rollout_epochs $epoch --log_to_wandb