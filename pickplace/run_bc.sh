for seed in 0 1 2 3
do
    python experiment_pickplace.py --seed ${seed} &
    sleep 20
done 
wait