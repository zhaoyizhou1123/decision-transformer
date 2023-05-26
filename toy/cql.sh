# python run_cql.py --epochs 100 --batch_size 4 --horizon 20 --data_file ./dataset/toy5_rev.csv --rate 6e-4 --arch '2' --env_path ./env/env_rev.txt --n_embd ${n_embd} --tradeoff_coef ${alpha} --tb_suffix $suffix
# python run_cql.py --epochs 5 --batch_size 4 --horizon 20 --data_file ./dataset/toy5.csv --rate 6e-4 --arch '10-10' --env_path ./env/env_bandit.txt

suffix=05251138

# for n_embd in 4 
# do
#     for alpha in 0
#     do
#         python run_cql.py --epochs 100 --batch_size 4 --horizon 20 --data_file ./dataset/toy5_rev.csv --rate 0.6 --arch '' --env_path ./env/env_rev.txt --n_embd ${n_embd} --tradeoff_coef ${alpha} --tb_suffix $suffix
#     done
# done

for lr in 1.5 
do
    for batch in 200
    do
        python run_cql.py --epochs 50000 --batch_size $batch --horizon 20 --data_file ./dataset/toy5_rev.csv --rate $lr --arch '' --env_path ./env/env_rev.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
    done
done