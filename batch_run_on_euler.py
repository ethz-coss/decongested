import os

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000

for GRID in ["uniform", "random", "braess"]:
    for NEXT_DESTINATION_METHOD in ["simple", "random", "work-commute"]:
        if GRID == "braess" and NEXT_DESTINATION_METHOD != "simple":
            continue
        for EXPLORATION_METHOD in ["random", "neighbours"]:
            for IOT_NODES in [True, False]:
                os.system(f'sbatch --mem-per-cpu=64G --time=48:00:00 --wrap="python dqn_grid_online.py {N_ITER} {NEXT_DESTINATION_METHOD} {EXPLORATION_METHOD} {SAVE_PATH} {GRID} {"--iot_nodes" if IOT_NODES else ""}"')
