import os

os.system("module load gcc/8.2.0 python/3.9.9")  # load appropriate modules for EULER cluster

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000
AGENT_IDS = True

for GRID in ["uniform", "random"]:
    for NEXT_DESTINATION_METHOD in ["simple", "one-way", "random", "work-commute"]:
        for EXPLORATION_METHOD in ["random", "neighbours"]:
            for IOT_NODES in [True, False]:
                os.system(f'sbatch --mem-per-cpu=64G --gpus=1 --gres=gpumem:32g --time=48:00:00 --wrap="python dqn_grid_train_centralized.py {N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} {"--iot_nodes" if IOT_NODES else ""} {"--with_agent_ids" if AGENT_IDS else ""}"')

# test
# os.system(f'sbatch --mem-per-cpu=64G --gpus=1 --gres=gpumem:32g --time=48:00:00 --wrap="python dqn_grid_train_centralized.py 1000 simple random {SAVE_PATH} uniform --iot_nodes --with_agent_ids"')
