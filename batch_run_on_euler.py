import os

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000

for NEXT_DESTINATION_METHOD in ["random", "work-commute"]:
    for EXPLORATION_METHOD in ["random", "neighbours"]:
        for IOT_NODES in [True, False]:
            os.system(f'sbatch '
                      f'--mem-per-cpu=8G '
                      f'--time=24:00:00 '
                      f'--wrap="python dqn_grid_online.py '
                      f'{N_ITER} '
                      f'{NEXT_DESTINATION_METHOD} '
                      f'{EXPLORATION_METHOD} '
                      f'{IOT_NODES} '
                      f'{SAVE_PATH}')
