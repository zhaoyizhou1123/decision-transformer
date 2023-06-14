# python run_cql.py --epochs 100 --batch_size 4 --horizon 20 --data_file ./dataset/toy5_rev.csv --rate 6e-4 --arch '2' --env_path ./env/env_rev.txt --n_embd ${n_embd} --tradeoff_coef ${alpha} --tb_suffix $suffix
# python run_cql.py --epochs 5 --batch_size 4 --horizon 20 --data_file ./dataset/toy5.csv --rate 6e-4 --arch '10-10' --env_path ./env/env_bandit.txt

suffix=06121834
env=./env/env_bandit.txt
data=./dataset/bandit_exp_hasha.csv
# batch=8000
lr=6e-4
scale=1


# for n_embd in 4 
# do
#     for alpha in 0
#     do
#         python run_cql.py --epochs 500 --batch_size 2000 --data_file ./dataset/rev_exp.csv --rate 0.05 --arch '' --env_path ./env/env_rev.txt --n_embd ${n_embd} --tradeoff_coef ${alpha} --tb_suffix $suffix
#         python run_cql.py --epochs 500 --batch_size 2000 --data_file ./dataset/bandit_exp.csv --rate 0.05 --arch '' --env_path ./env/env_bandit.txt --n_embd ${n_embd} --tradeoff_coef ${alpha} --tb_suffix $suffix
#         python run_cql.py --epochs 500 --batch_size 2000 --data_file ./dataset/toy6.csv --rate 0.05 --arch '' --env_path ./env/env_bandit.txt --n_embd ${n_embd} --tradeoff_coef ${alpha} --tb_suffix $suffix
#     done
# done



# for arch in '' '10' '10-10'
# do
#     for action_repeat in 10 1
#     do 
#         for scale in 1 10
#         do
#         # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '2' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
#         # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '5' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
#         # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '2-2' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
#         # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '5-5' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
#             python run_cql.py --epochs 500 --batch_size $batch --data_file ./dataset/3s-h20_exp.csv --rate $lr --arch $arch --env_path ./env/env_3s_h20.txt --tradeoff_coef 0 --tb_suffix $suffix --action_repeat ${action_repeat} --scale $scale
#         done
#     done
# done

for action_repeat in 1 10
do
    for batch in 8000
    do
        for n_embd in 1 2 4
        do
        # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '2' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
        # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '5' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
        # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '2-2' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
        # python run_cql.py --epochs 100 --batch_size $batch --data_file ./dataset/3s_rand.csv --rate $lr --arch '5-5' --env_path ./env/env_3s_rand.txt --n_embd 4 --tradeoff_coef 0 --tb_suffix $suffix
            python run_cql.py --epochs 200 --batch_size $batch --data_file $data --rate $lr --arch '' --env_path $env --tradeoff_coef 0 --tb_suffix $suffix --repeat ${action_repeat} --scale $scale --n_embd ${n_embd}
            # for arch in '1' '2' '5'
            # do
            #     python run_cql.py --epochs 200 --batch_size $batch --data_file $data --rate $lr --arch $arch --env_path $env --tradeoff_coef 0 --tb_suffix $suffix --repeat ${action_repeat} --scale $scale --n_embd ${n_embd}
            # done
        done
    done
done