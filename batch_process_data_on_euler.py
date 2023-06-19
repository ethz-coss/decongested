import os
import numpy as np

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
INTERNAL_SAVE_PATH = "/cluster/home/ccarissimo/decongested/processed_data"
N_ITER = 400000

# testing stationary
for GRID in ["uniform", "random"]:
    for NEXT_DESTINATION_METHOD in ["one-way", "work-commute"]:
        for EXPLORATION_METHOD in ["random"]:
            for IOT_NODES in [True]:
                for CENTRALIZED_RATIO in np.linspace(0, 1, 21):
                    for AGENT_IDS in [True, False]:
                        os.system(f'sbatch --mem-per-cpu=32G --gpus=1 --time=04:00:00 --wrap="python process_evaluations_data.py '
                                  f'{N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} '
                                  f'{CENTRALIZED_RATIO} {INTERNAL_SAVE_PATH}  {"--iot_nodes" if IOT_NODES else ""} '
                                  f'{"--with_agent_ids" if AGENT_IDS else ""}" '
                                  f'--job-name=grid-{GRID}-dex-{NEXT_DESTINATION_METHOD}-exp-{EXPLORATION_METHOD}-'
                                  f'iot-{IOT_NODES}-ratio-{CENTRALIZED_RATIO}')


# testing non-stationary
# INTERNAL_SAVE_PATH = "/cluster/home/ccarissimo/decongested/processed_non_stationary_data"
# for GRID in ["uniform"]:
#     for NEXT_DESTINATION_METHOD in ["one-way", "work-commute"]:
#         for EXPLORATION_METHOD in ["random", "neighbours"]:
#             for IOT_NODES in [True, False]:
#                 for CENTRALIZED_RATIO in [0, 0.1, 0.5, 0.9, 1]:
#                     os.system(f'sbatch --mem-per-cpu=32G --gpus=1 --time=04:00:00 --wrap="python process_non_stationary_evaluations.py '
#                               f'{N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} '
#                               f'{CENTRALIZED_RATIO} {INTERNAL_SAVE_PATH}  {"--iot_nodes" if IOT_NODES else ""}" '
#                               f'--job-name=grid-{GRID}-dex-{NEXT_DESTINATION_METHOD}-exp-{EXPLORATION_METHOD}-'
#                               f'iot-{IOT_NODES}-ratio-{CENTRALIZED_RATIO}')

# test
# os.system(f'sbatch --mem-per-cpu=32G --time=48:00:00 --wrap="python pre_process_data.py 400000 simple random /cluster/scratch/ccarissimo/decongested uniform 0 /cluster/home/ccarissimo/decongested/test_process --iot_nodes"')
