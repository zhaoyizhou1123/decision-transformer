# Learn mdp

ckpt_root=./checkpoint/
ckpt_name=test_reward
# tb_path=None
r_loss_weight=0.5
data_file=./dataset/maze_moredata.dat
mdp_epoch=100

for d_arch in '256-256'
do
    python scripts/learn_mdp_main.py --ckpt_root ${ckpt_root} --ckpt_name ${ckpt_name} --r_loss_weight ${r_loss_weight} --d_arch ${d_arch} --data_file ${data_file} --mdp_epoch ${mdp_epoch}
done