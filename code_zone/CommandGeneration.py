# -*- coding: utf-8 -*-
import os
import time
import numpy as np

if not os.path.exists('output'):
    os.makedirs('output')

# %%
EARLY_STOPPING = [0, 1]  # 0, 1
BATCH_SIZE = [25, 50]
PATCH_SIZE = [256, 512]

# %%
# making sure there is no repetition in the parameters lists
EARLY_STOPPING = np.unique(EARLY_STOPPING).tolist()
BATCH_SIZE = np.unique(BATCH_SIZE).tolist()
PATCH_SIZE = np.unique(PATCH_SIZE).tolist()

# %%
'''
###########
Careful about the batch size. High value can cause the GPU memory to be full an the job to get terminated
###########
'''

# OUTPUT_PATH = 'output/results_' + time.strftime("%d%m-%H%M%S") + '/'

with open("output\Runner.py", "a") as f:
    f.write('import os' + '\n')
    f.write('\n')

    TotalCommands = 0
    for BS in BATCH_SIZE:
        for PS in PATCH_SIZE:
            for ES in EARLY_STOPPING:
                f.write('os.system(\'python3 main.py --EARLY_STOPPING '
                        + str(ES) + ' --Patch_Size '
                        + str(PS) + ' --Batch_Size '
                        + str(BS) + '\')' + '\n')
                TotalCommands += 1

print(TotalCommands)

# %%
# Splitting the commands in Runner.py into N Runner_x.py files (because there is a 24 hours limit on the cluster and only a handful of commands can run in that time. Further, we can run 4 simultanous jobs.)
import numpy as np

N = 4  # number of Runner_x.py (N should be set manually depending on the TotalCommands)
x = np.floor(np.linspace(0, TotalCommands, N + 1)).astype('int32')

with open("output/Runner.py", "r") as f1:
    lines = list(f1)

for i in range(len(x) - 1):
    with open("output/Runner" + str(i + 1) + ".py", "w") as f2:
        f2.write('import os' + '\n\n')
        try:
            for line in np.arange(x[i] + 2, x[i + 1] + 2):
                f2.write(lines[line])
        except:
            pass
