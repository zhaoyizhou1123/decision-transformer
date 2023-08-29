import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
# mlp = np.array([73.5080, 75.4309, 72.9874, 97.0514, 73.8498])
# mlp_e = np.array([17.4243, 21.8219, 32.6713, 2.1206, 10.4174])
# cql = np.array([62.5855, 81.9166, 85.1735, 82.3918, 82.1984])
# cql_e = np.array([33.9725, 8.1303, 8.0752, 21.1405, 13.2079])
mlp = np.array([70.7961, 87.9667, 79.2540, 89.8095, 90.7435])
mlp_e = np.array([29.8575, 13.8911, 15.1238, 11.9173, 1.3307])
cql = np.array([82.6306, 81.1179, 80.8299, 74.5368, 70.0115])
cql_e = np.array([9.5577, 13.8389, 10.8767, 9.3758, 17.0877])

xticks = ('128', '256', '512','1024', '2048')
plt.xticks(np.arange(1,6), xticks)
plt.xlabel('Network width')
plt.ylabel('Average return')

plt.errorbar(x, mlp, mlp_e, linestyle='None', marker='o', markersize=5., label='MLP')
plt.errorbar(x+0.2, cql, cql_e, linestyle='None', marker='s', markersize=5., label='CQL')
plt.legend()
plt.title('Rollout Data, depth=4')

plt.savefig("stitch-mlp,cql,d4.png")