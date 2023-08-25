# Used to clean data

# ratio=0
# arch="200-200-200"
# target_dirs="./cql_log/default_2023_08_21_*--arch${arch}--cql"
# output_file=${base_dir}/arch${arch}.out
# echo -n > ${output_file}
# for dir in ${target_dirs}
# do
#     echo $dir
#     grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
#     grep average_return $dir/debug.log >> ${output_file}
# done

# base_dir=./backup/stitch-cql-20230823-ratio0
# mkdir ${base_dir}
# algo=cql
# for width in 4096
# do
#     arch="${width}-${width}"
#     target_dirs="./${algo}_log/default_2023_08_23_*--arch${arch}--${algo}"
#     output_file=${base_dir}/arch${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
#         grep average_return $dir/debug.log >> ${output_file}
#     done
#     arch="${width}-${width}-${width}-${width}"
#     target_dirs="./${algo}_log/default_2023_08_24_*--arch${arch}--${algo}"
#     output_file=${base_dir}/arch${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
#         grep average_return $dir/debug.log >> ${output_file}
#     done
# done

# base_dir=./backup/stitch-mlp-20230824-rolloutonly
# mkdir ${base_dir}
# algo=mlp
# for width in 4096 2048 1024 512 256 128
# do
#     arch="${width}-${width}"
#     target_dirs="./${algo}_log/default_2023_08_24_*--arch${arch}--${algo}"
#     output_file=${base_dir}/arch${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
#         grep average_return $dir/debug.log >> ${output_file}
#     done
#     arch="${width}-${width}-${width}-${width}"
#     target_dirs="./${algo}_log/default_2023_08_24_*--arch${arch}--${algo}"
#     output_file=${base_dir}/arch${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -P '"algo"|"seed"|"arch"' $dir/variant.json >> ${output_file}
#         grep average_return $dir/debug.log >> ${output_file}
#     done
# done

base_dir=./backup/stitch-mlp-gaussian-20230824-rolloutonly
mkdir ${base_dir}
algo=mlp
target_file="maze/backup/stitch-mlp-gaussian-20230824-rolloutonly/all.out"
for w in 128 256 512 1024 2048 4096
do
    arch="${w}-${w}"
    output_file=${base_dir}/arch${arch}.out
    # echo -n > ${output_file}
    for dir in ${target_dirs}
    do
        echo $dir
        grep -Po '"seed": [0-9]|"arch": "[0-9 -]*"' $dir/hyper_param.json >> ${output_file}
        grep -P episode_reward $dir/consoleout_backup.txt >> ${output_file}
    done
done

# ratio=0.25
# target_dirs=./cql_log/default_2023_08_16_21_*
# output_file=${base_dir}/ratio${ratio}.out
# echo -n > ${output_file}
# for dir in ${target_dirs}
# do
#     echo $dir
#     grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
#     grep average_return $dir/debug.log >> ${output_file}
# done

# ratio=0.5
# target_dirs=./cql_log/default_2023_08_16_23_*
# output_file=${base_dir}/ratio${ratio}.out
# echo -n > ${output_file}
# for dir in ${target_dirs}
# do
#     echo $dir
#     grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
#     grep average_return $dir/debug.log >> ${output_file}
# done

# ratio=0.75
# target_dirs=./cql_log/default_2023_08_17_00_*
# output_file=${base_dir}/ratio${ratio}.out
# echo -n > ${output_file}
# for dir in ${target_dirs}
# do
#     echo $dir
#     grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
#     grep average_return $dir/debug.log >> ${output_file}
# done

# ratio=1
# target_dirs=./cql_log/default_2023_08_17_02_*
# output_file=${base_dir}/ratio${ratio}.out
# echo -n > ${output_file}
# for dir in ${target_dirs}
# do
#     echo $dir
#     grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
#     grep average_return $dir/debug.log >> ${output_file}
# done

# find ./cql_log
# grep -h average_return cql_log/default_2023_08_14_21_58_*/debug.log