for seed in 0 1 2 3
do
    python run_dt_pickplace.py --num_workers 2 --seed ${seed} --ctx 10 --epoch 500 &
    sleep 20
done 
wait

# python run_dt_pickplace.py --num_workers 4 --seed 0 --ctx 5