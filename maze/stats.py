import pandas as pd
import numpy as np
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('data', type=float, nargs='*')

# args = parser.parse_args()
# stats =args.data

data_per_exp = int(50)
for ratio in ['0', '0.25','0.5','0.75','1']:
    file = f"./backup/stitch-cql-20230817/ratio{ratio}.out"
    with open(file,"r") as f:
        lines = f.readlines()
    data_lines = []
    for line in lines:
        if line[0] == 'a': # average return
            line = line.split()
            data = float(line[-1])
            data_lines.append(data)

    stats = [data_lines[data_per_exp-1], data_lines[2*data_per_exp-1], data_lines[3*data_per_exp-1], data_lines[4*data_per_exp-1]]
    print(f"Stats: {stats}")


    print(f"Ratio {ratio}. Mean: {np.mean(stats)}, std: {np.std(stats)}")
