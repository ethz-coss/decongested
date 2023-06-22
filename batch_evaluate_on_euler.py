import os
import numpy as np

os.system("module load gcc/8.2.0 python/3.9.9")  # load appropriate modules for EULER cluster

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000

# for GRID in ["uniform", "random"]:
#     for NEXT_DESTINATION_METHOD in ["one-way", "work-commute"]:
#         for EXPLORATION_METHOD in ["random"]:
#             for IOT_NODES in [True]:
#                 for RATIO in np.linspace(0, 1, 21):
#                     for AGENT_IDS in [True, False]:
#                         os.system(f'sbatch '
#                                   f'--mem-per-cpu=16G '
#                                   f'--gpus=1 '
#                                   f'--time=12:00:00 '
#                                   f'--job-name=evaluate_{GRID}_{NEXT_DESTINATION_METHOD}_{RATIO}_{AGENT_IDS} '
#                                   f'--wrap="python dqn_grid_evaluate_online.py '
#                                   f'{N_ITER} '
#                                   f'{NEXT_DESTINATION_METHOD} '
#                                   f'{EXPLORATION_METHOD} '
#                                   f'{SAVE_PATH} '
#                                   f'{GRID} '
#                                   f'{RATIO} '
#                                   f'--iot_nodes '
#                                   f'{"--with_ids" if AGENT_IDS else ""}"'
#                                   )

AGENT_IDS = True
EXPERIMENT_NAME = "random_random_transient"
for GRID in ["random"]:
    for NEXT_DESTINATION_METHOD in ["one-way", "work-commute"]:
        for EXPLORATION_METHOD in ["random"]:
            for IOT_NODES in [True]:
                for RATIO in [0, 0.1, 0.5, 0.9, 1]:
                    for TRAIN in [True]:
                        for NON_STATIONARY in [True]:
                            os.system(f'sbatch '
                                      f'--mem-per-cpu=16G '
                                      f'--gpus=1 '
                                      f'--time=24:00:00 '
                                      f'--job-name=evaluate_non_stationary_{NEXT_DESTINATION_METHOD}_{RATIO} '
                                      f'--wrap="python dqn_grid_evaluate_online.py '
                                      f'{N_ITER} '
                                      f'{NEXT_DESTINATION_METHOD} '
                                      f'{EXPLORATION_METHOD} '
                                      f'{SAVE_PATH} '
                                      f'{GRID} '
                                      f'{RATIO} '
                                      f'{EXPERIMENT_NAME} '
                                      f'--iot_nodes '
                                      f'--with_ids '
                                      f'--non_stationary '
                                      f'--train"'
                                      )

# test
# os.system(f'sbatch --mem-per-cpu=16G --gpus=1 --time=04:00:00 --wrap="python dqn_grid_evaluate_online.py {N_ITER} random random {SAVE_PATH} uniform 0.5 --iot_nodes"')
