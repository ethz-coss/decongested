import os

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested/federated"
N_EPISODES = 400
N_ITER = 6400
IOT_NODES = False
EXPLORATION_METHOD = "random"
NEXT_DESTINATION_METHOD = "one-way"
REPETITIONS = 10

for GRID in ["initial", "braess"]:
    for AVERAGING_METHOD in ["None", "agents_at_base_node", "agents_not_at_base_node", "all_agents"]:
        for i in range(REPETITIONS):
            os.system(f'sbatch --mem-per-cpu=64G --time=48:00:00 --wrap="python dqn_grid_online.py {N_EPISODES} {N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {AVERAGING_METHOD} {SAVE_PATH} {GRID} --train "')

# test
# sbatch --mem-per-cpu=64G --time=48:00:00 --wrap="python dqn_grid_online.py 1000 simple random /cluster/scratch/ccarissimo/decongested uniform --iot_nodes"
