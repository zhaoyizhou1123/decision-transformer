import numpy as np

np.random.seed(0)

def f():
    for _ in range(10):
        print(np.random.choice(2))

f()

