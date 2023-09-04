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

base_dir=./backup/stitch-cql-20230901-ratio1-long
mkdir ${base_dir}
algo=cql
for width in 1024
do
    arch="${width}-${width}"
    target_dirs="./${algo}_log/default_2023_08_30_*--arch${arch}--${algo}"
    output_file=${base_dir}/arch${arch}.out
    echo -n > ${output_file}
    for dir in ${target_dirs}
    do
        echo $dir
        grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
        grep average_return $dir/debug.log >> ${output_file}
    done
    # arch="${width}-${width}-${width}-${width}"
    # target_dirs="./${algo}_log/default_2023_08_29_*--arch${arch}--${algo}"
    # output_file=${base_dir}/arch${arch}.out
    # echo -n > ${output_file}
    # for dir in ${target_dirs}
    # do
    #     echo $dir
    #     grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
    #     grep average_return $dir/debug.log >> ${output_file}
    # done
done

# base_dir=./backup/stitch-mlp-20230828-rolloutonly
# mkdir ${base_dir}
# algo=mlp
# for width in 4096 2048 1024 512 256 128
# do
#     arch="${width}-${width}"
#     target_dirs="./${algo}_log/default_2023_08_25_*--arch${arch}--${algo}"
#     output_file=${base_dir}/arch${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
#         grep average_return $dir/debug.log >> ${output_file}
#     done
#     arch="${width}-${width}-${width}-${width}"
#     target_dirs="./${algo}_log/default_2023_08_25_*--arch${arch}--${algo}"
#     output_file=${base_dir}/arch${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -P '"algo"|"seed"|"arch"' $dir/variant.json >> ${output_file}
#         grep average_return $dir/debug.log >> ${output_file}
#     done
# done

# base_dir=./backup/stitch-mlp-gaussian-20230828-rolloutonly
# mkdir ${base_dir}
# algo=mlp
# for width in 4096 2048 1024 512 256 128
# do
#     arch="${width}-${width}"
#     target_dirs="logs/pointmaze/stitch-mlp-gaussian/rcsl/timestamp_23-0825-1*&${arch}-s*/record"
#     output_file=${base_dir}/${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -Po '"seed": [0-9]|"arch": "[0-9 -]*"' $dir/hyper_param.json >> ${output_file}
#         grep -P episode_reward $dir/consoleout_backup.txt >> ${output_file}
#     done
# done
# for width in 4096 2048 1024 512 256 128
# do
#     arch="${width}-${width}-${width}-${width}"
#     target_dirs="logs/pointmaze/stitch-mlp-gaussian/rcsl/timestamp_23-0825-1*&${arch}-s*/record"
#     output_file=${base_dir}/${arch}.out
#     echo -n > ${output_file}
#     for dir in ${target_dirs}
#     do
#         echo $dir
#         grep -Po '"seed": [0-9]|"arch": "[0-9 -]*"' $dir/hyper_param.json >> ${output_file}
#         grep -P episode_reward $dir/consoleout_backup.txt >> ${output_file}
#     done
# done

base_dir=./backup/rcsl-mlp-expert-20230901-long
mkdir ${base_dir}
algo=rcsl-mlp
net=mlp
for width in 1024
do
    arch="${width}-${width}"
    target_dirs="./${algo}_log/default_2023_08_31_*--arch${arch}--${net}"
    output_file=${base_dir}/arch${arch}.out
    echo -n > ${output_file}
    for dir in ${target_dirs}
    do
        echo $dir
        grep -P 'offline_ratio|"seed"|"arch"' $dir/variant.json >> ${output_file}
        grep average_return $dir/debug.log >> ${output_file}
    done
    # arch="${width}-${width}-${width}-${width}"
    # target_dirs="./${algo}_log/default_2023_08_28_*--arch${arch}--${net}"
    # output_file=${base_dir}/arch${arch}.out
    # echo -n > ${output_file}
    # for dir in ${target_dirs}
    # do
    #     echo $dir
    #     grep -P '"algo"|"seed"|"arch"' $dir/variant.json >> ${output_file}
    #     grep average_return $dir/debug.log >> ${output_file}
    # done
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