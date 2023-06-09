import pickle

out_file = 'output/ret_group8.out'

with open(out_file, 'rb') as f:
    rtgs, clusters = pickle.load(f)

for ind in range(len(rtgs)):
    print(f"Trajectory {ind}")
    print(f"Rtgs {rtgs[ind]}")
    print(f"Clusters {clusters[ind]}")
    print("--------------------")
