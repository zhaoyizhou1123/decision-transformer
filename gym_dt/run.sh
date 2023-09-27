for seed in 0 1 2 3
do 
    python experiment_maze.py --seed ${seed} &
    sleep 60
done
wait