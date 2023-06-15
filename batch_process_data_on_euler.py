import os


SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
INTERNAL_SAVE_PATH = "/cluster/home/ccarissimo/decongested/processed_data"
N_ITER = 400000

for GRID in ["uniform", "random"]:
    for NEXT_DESTINATION_METHOD in ["simple", "one-way", "random", "work-commute"]:
        for EXPLORATION_METHOD in ["random", "neighbours"]:
            for IOT_NODES in [True, False]:
                for CENTRALIZED_RATIO in [0, 0.05, 0.5, 1]:
                    os.system(f'sbatch --mem-per-cpu=32G --time=48:00:00 --wrap="python pre_process_data.py {N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} {INTERNAL_SAVE_PATH} {"--iot_nodes" if IOT_NODES else ""}"')
