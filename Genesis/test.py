import numpy as np

b = 12
a = np.arange(b).reshape((2, 6))
def get(idx : int):
        # print(a)
        rows, cols = a.shape
        run = idx // cols
        # print(run)
        run_id = idx - sum(len(x) for x in a[:run,])
        # print(run_id)
        sample = a[run][run_id]
        print(sample)
    
for i in range(b):
    get(i)