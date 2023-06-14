import os

os.system("module load gcc/8.2.0 python/3.9.9")  # load appropriate modules for EULER cluster

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000

for GRID in ["uniform", "random"]:
    for NEXT_DESTINATION_METHOD in ["simple", "one-way", "random", "work-commute"]:
        for EXPLORATION_METHOD in ["random", "neighbours"]:
            for IOT_NODES in [True, False]:
                for RATIO in [0, 0.1, 0.5, 1]:
                    os.system(f'sbatch --mem-per-cpu=16G --gpus=1 --time=04:00:00 --wrap="python dqn_grid_evaluate_online.py {N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} {RATIO} {"--iot_nodes" if IOT_NODES else ""}"')

# test
# os.system(f'sbatch --mem-per-cpu=16G --gpus=1 --time=04:00:00 --wrap="python dqn_grid_evaluate_online.py {N_ITER} random random {SAVE_PATH} uniform 0.5 --iot_nodes"')
