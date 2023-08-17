# Used to clean data

base_dir=./backup/stitch-cql-20230817
mkdir ${base_dir}



ratio=0
target_dirs=./cql_log/default_2023_08_16_19_*
output_file=${base_dir}/ratio${ratio}.out
echo -n > ${output_file}
for dir in ${target_dirs}
do
    echo $dir
    grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
    grep average_return $dir/debug.log >> ${output_file}
done

ratio=0.25
target_dirs=./cql_log/default_2023_08_16_21_*
output_file=${base_dir}/ratio${ratio}.out
echo -n > ${output_file}
for dir in ${target_dirs}
do
    echo $dir
    grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
    grep average_return $dir/debug.log >> ${output_file}
done

ratio=0.5
target_dirs=./cql_log/default_2023_08_16_23_*
output_file=${base_dir}/ratio${ratio}.out
echo -n > ${output_file}
for dir in ${target_dirs}
do
    echo $dir
    grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
    grep average_return $dir/debug.log >> ${output_file}
done

ratio=0.75
target_dirs=./cql_log/default_2023_08_17_00_*
output_file=${base_dir}/ratio${ratio}.out
echo -n > ${output_file}
for dir in ${target_dirs}
do
    echo $dir
    grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
    grep average_return $dir/debug.log >> ${output_file}
done

ratio=1
target_dirs=./cql_log/default_2023_08_17_02_*
output_file=${base_dir}/ratio${ratio}.out
echo -n > ${output_file}
for dir in ${target_dirs}
do
    echo $dir
    grep -P 'offline_ratio|"seed"' $dir/variant.json >> ${output_file}
    grep average_return $dir/debug.log >> ${output_file}
done

# find ./cql_log
# grep -h average_return cql_log/default_2023_08_14_21_58_*/debug.log