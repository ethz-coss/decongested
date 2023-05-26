import os

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000

for NEXT_DESTINATION_METHOD in ["random", "work-commute"]:
    for EXPLORATION_METHOD in ["random", "neighbours"]:
        for IOT_NODES in [True, False]:
            os.system(f'sbatch --mem-per-cpu=8G --time=24:00:00 --wrap="python dqn_grid_online.py {N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {IOT_NODES} {SAVE_PATH}')
