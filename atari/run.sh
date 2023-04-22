# Decision Transformer (DT)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done

# Behavior Cloning (BC)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done

# Own cmds
python run_dt_atari.py --epochs 5 --model_type 'reward_conditioned' --num_steps 5000 --num_buffers 50 --game 'Breakout' --data_dir_prefix './at_replay'
python run_dt_atari.py --epochs 1 --num_steps 100 --num_buffers 1
# hyak cmds
salloc -p gpu-a40 -A simondu --time=1:00:00 --nodes=1 --cpus-per-task=1 --mem=16G --gres=gpu:1
salloc -p gpu-rtx6k -A cse --time=0:5:00 --nodes=1 --cpus-per-task=1 --mem=16G
salloc -p gpu-2080ti -A cse --time=0:2:00 --nodes=1 --cpus-per-task=1 --mem=16G

pip install atari-py pyprind absl-py gin-config gym tqdm blosc git+https://github.com/google/dopamine.git