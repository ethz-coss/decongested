import os


SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
INTERNAL_SAVE_PATH = "/cluster/home/ccarissimo/decongested/processed_data"
N_ITER = 400000

for GRID in ["uniform", "random"]:
    for NEXT_DESTINATION_METHOD in ["simple", "one-way", "random", "work-commute"]:
        for EXPLORATION_METHOD in ["random", "neighbours"]:
            for IOT_NODES in [True, False]:
                for CENTRALIZED_RATIO in [0, 0.05, 0.5, 1]:
                    os.system(f'sbatch --mem-per-cpu=32G --gpus=1 --time=04:00:00 --wrap="python pre_process_data.py '
                              f'{N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} '
                              f'{CENTRALIZED_RATIO} {INTERNAL_SAVE_PATH}  {"--iot_nodes" if IOT_NODES else ""}" '
                              f'--job-name=grid-{GRID}-dex-{NEXT_DESTINATION_METHOD}-exp-{EXPLORATION_METHOD}-'
                              f'iot-{IOT_NODES}-ratio-{CENTRALIZED_RATIO}')

# test
# os.system(f'sbatch --mem-per-cpu=32G --time=48:00:00 --wrap="python pre_process_data.py 400000 simple random /cluster/scratch/ccarissimo/decongested uniform 0 /cluster/home/ccarissimo/decongested/test_process --iot_nodes"')
