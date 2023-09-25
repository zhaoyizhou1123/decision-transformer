import pandas as pd
import numpy as np
import argparse

from collections import defaultdict

# parser = argparse.ArgumentParser()
# parser.add_argument('data', type=float, nargs='*')

# args = parser.parse_args()
# stats =args.data

def stats_cql(debug=False):
    data_per_exp = int(50)
    for depth in [2,4]:
        for width in ['128','256','512','1024','2048']:
            arch = (width+'-') * depth
            arch = arch[:-1]
            file = f"./backup/cql-expert-20230829-ratio1/arch{arch}.out"
            with open(file,"r") as f:
                lines = f.readlines()
            data_lines = []
            for line in lines:
                if line[0] == 'a': # average return
                    line = line.split()
                    data = float(line[-1])
                    data_lines.append(data)

            stats = [data_lines[data_per_exp-1], data_lines[2*data_per_exp-1], data_lines[3*data_per_exp-1], data_lines[4*data_per_exp-1]]

            if debug:
                print(f"Stats: {stats}")


            print(f"{'CQL':<18}{arch:<20}{np.mean(stats):10.4f}{np.std(stats):19.4f}")

def stats_mlp(debug=False):
    data_per_exp = int(51)
    for depth in [2,4]:
        for width in ['128','256','512','1024','2048','4096']:
            arch = (width+'-') * depth
            arch = arch[:-1]
            file = f"./backup/stitch-mlp-20230828-rolloutonly/arch{arch}.out"
            with open(file,"r") as f:
                lines = f.readlines()
            data_lines = []
            for line in lines:
                if line[0] == 'a': # average return
                    line = line.split()
                    data = float(line[-1])
                    data_lines.append(data)

            stats = [data_lines[data_per_exp-1], data_lines[2*data_per_exp-1], data_lines[3*data_per_exp-1], data_lines[4*data_per_exp-1]]
            if debug:
                print(f"Stats: {stats}")


            print(f"MLP{'':>15}{arch:<20}{np.mean(stats):10.4f}{np.std(stats):19.4f}")

def stats_rcsl_mlp(debug=False):
    data_per_exp = int(102)
    for depth in [2,4]:
        for width in ['128','256','512','1024','2048','4096']:
            arch = (width+'-') * depth
            arch = arch[:-1]
            file = f"./backup/rcsl-mlp-20230828/arch{arch}.out"
            with open(file,"r") as f:
                lines = f.readlines()
            data_lines = []
            for line in lines:
                if line[0] == 'a': # average return
                    line = line.split()
                    data = float(line[-1])
                    data_lines.append(data)

            stats = [data_lines[data_per_exp-1], data_lines[2*data_per_exp-1], data_lines[3*data_per_exp-1], data_lines[4*data_per_exp-1]]
            if debug:
                print(f"Stats: {stats}")


            print(f"MLP{'':>15}{arch:<20}{np.mean(stats):10.4f}{np.std(stats):19.4f}")

def stats_mlp_gaussian2(debug=False):
    line_per_exp = int(202)
    num_exp = int(4)
    for depth in [2,4]:
        for width in ['128','256','512','1024','2048','4096']:
            arch = (width+'-') * depth
            arch = arch[:-1]
            file = f"./backup/stitch-mlp-gaussian-20230828-rolloutonly/{arch}.out"
            with open(file,"r") as f:
                lines = f.readlines()
            arch_lines = [lines[line_per_exp*i] for i in range(num_exp)]
            arch_lines = [line.split('"')[3] for line in arch_lines] # '128-128', ...
            ret_lines = [lines[line_per_exp*i+line_per_exp-2] for i in range(num_exp)]
            ret_lines = [float(line.split("|")[2]) for line in ret_lines] # ele: avg_ret

            stats = ret_lines
            if debug:
                print(f"Stats: {stats}")


            print(f"{'MLP-Gauss':<18}{arch:<20}{np.mean(stats):10.4f}{np.std(stats):19.4f}")

def stats_mlp_gaussian():
    line_per_exp = int(102)
    num_exp = int(24)
    for depth in [2]:
        file = f"./backup/stitch-mlp-gaussian-20230824-rolloutonly/depth{depth}.out"
        with open(file,"r") as f:
            lines = f.readlines()
        arch_lines = [lines[line_per_exp*i] for i in range(num_exp)]
        arch_lines = [line.split('"')[3] for line in arch_lines] # '128-128', ...
        ret_lines = [lines[line_per_exp*i+line_per_exp-2] for i in range(num_exp)]
        ret_lines = [float(line.split("|")[2]) for line in ret_lines] # ele: avg_ret

        arch_data_dict = defaultdict(list)
        for arch_line, ret_line in zip(arch_lines, ret_lines):
            arch_data_dict[arch_line].append(ret_line)

        for width in ['128','256','512','1024','2048','4096']:
            arch = (width+'-') * depth
            arch = arch[:-1]
            stats = arch_data_dict[arch]
            # print(f"Stats: {stats}")

            print(f"MLP. Arch {arch}. Mean: {np.mean(stats):.4f}, std: {np.std(stats):.4f}")

debug=True
print(f"Algo{'':>14}Arch{'':>19}Mean{'':>15}Std{'':>15}")
# stats_mlp(True)
# stats_rcsl_mlp()
stats_cql(debug)
# stats_mlp_gaussian2()