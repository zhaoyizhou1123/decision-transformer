# Learn mdp

ckpt_root=./checkpoint/
ckpt_name=layer4_stateonly_1e6
# tb_path=None
r_loss_weight=0
data_file=./dataset/maze_1e6.dat
mdp_epoch=20
train_type=default

for d_arch in '128-128-128-128'
do
    python scripts/learn_mdp_main.py --ckpt_root ${ckpt_root} --ckpt_name ${ckpt_name} --r_loss_weight ${r_loss_weight} --d_arch ${d_arch} --data_file ${data_file} --mdp_epoch ${mdp_epoch} --train_type ${train_type} --log_to_wandb
done