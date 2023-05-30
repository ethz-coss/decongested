my motivation is in the ground

I need to resume coding and development work

I should set myself some simple easy tasks that can be done that will move things forward



goals

- Two cities test, two separate populations of drivers that eventually interact
- shared exploration, novelty search
- learning on non-grid environments, arbitrary street networks, BPR functions
- estimation of socially optimal solutions for comparison
- dynamic changing of the street networks, an open world with many agents 
- Live online learning
- being able to test all these things modularly, such that any combination is available for comparison



descriptions

- shared exploration
  - epsilon greedy, 
    - when exploring, rather than randomly, they ask neighbours
    - an average is taken of the q-values of all neighbours
    - the agents picks the argmax of the average
- live online learning
  - no episodes
  - After an agent has completed an OD pair it is assigned a new OD pair
    - OD pairs may be assigned randomly
    - OD pairs may be fixed for agents (home, work) (warehouseA, warehouseB)



tasks

- live online learning
  - ~~remove episodes~~
    - ~~move training/optimization step to happen every x timesteps~~
  - figure out how to create the reward structure such that agents know to go to their destination
    - the issues arise because the travel time is the only reward, and their travel time does not reduce if they immediately receive a new destination and keep driving
    - one work around could be to give the agents at the final state the chance to pick an action with 0 reward, so 0 cost, for t rounds in the final state
    - ~~implement reward 0 for the destination state, while all other states get reward -travel_time~~
  - ~~random destination assignment~~
    - ~~whenever an agent reaches the final state, wait t rounds, then assign a new final state~~
    - ~~single driver has a hard time learning~~
      - ~~randomly assigned destination may not be trained on before, may not appear frequently~~
      - ~~exploration rate decay interaction with new task assignment needs to be adjusted~~
      - ~~solved by training for longer time with higher exploration rate~~
  - fixed od pairs
    - ~~whenever an agent reaches the final state, wait t round, then assign the other destination~~
    - ~~implement simple so that agents go from initial --> final, then final --> initial~~
    - ~~start with a dictionary of commuter OD pairs for each driver and alternate between them when drivers reach the destination~~
  - realistically, the system will have combinations of both random and fixed, and mutli step fixed routes, so each agent should be able to have its settings specified independently
- code refactoring/cleaning
  - ~~rename final_states to destinations~~
  - ~~save driver NN weights in easy to re-use files/folder~~
  - ~~save experiment results in easy to access folder and well-named files~~
  - save tested grid environment for easy loading and plotting
- shared exploration
  - take epsilon greedy agents
    - when exploring, instead of picking a random action, pick the argmax of the average q-values of neighbouring agents
- adding noise to weights for better generalisation, training and testing
  - find some relevant literature that may have tried this already
- Pre-trained single agents that know how to get from O to D, trained independently, and loaded into the model
  - single agent, start-to-finish commute, 4x4 grid, 100000 iterations, manually decayed epsilon (0.9, 0.05, n_iter)
  - ~~single agent, random destinations, 4x4 grid, 100000 iterations, manually decayed epsilon (0.9, 0.05, n_iter)~~
    - ~~Tested training, but agent does not successfully learn to get to all destinations, in fact, the agent does rather poorly~~
    - ~~Try increase batch size (64), increase buffer max_memory (10000), increase iterations (200000), keep exploration high (0.9 decayed to 0.33)~~
  - ~~test single agent trained model above with 100 copies of itself in training environment~~
- addition of a new agent that is "dumb" to see how congestion is affected
  - Steps:
    - take agent trained on its own with random destinations --> copy to 100 agents
    - run 100 agents with fixed commuter OD pairs until performance metric is reduced
    - remove M agents, replace them with "dumb" agents, and see how long it takes system to reach original performance in two different settings
      - independent agents, learn independently with local information and **random** exploration
      - collective agents, learn independently with local information and **shared** exploration
- how will system performance be calculated when the learning is live online?
  - ~~calculate the average time it takes agents to travel trips, normalized by the distance~~
    - ~~use manhattan distance to normalize the average travel times for trips~~
    - ~~store the destinations of agents at the completion of each trip, such that the distance can be calculated after the simulation~~
- ~~experiments~~
  - ~~run a comparison of the shared exploration and random exploration agents, starting from a pretrained agent~~
- Bug Fixes:
  - ~~fastest agents are now always the ones that reach their destination, because their travel time is set to 0: must keep track of trip travel time separately from total travel time~~
- Create an inference only script to test the learned behaviour of agents, generate a video, plot some results and statistics
- Figure out why model pickle files are so huge, and reduce their size
  - probably the memory buffer, but could be other things?
- how can `learnability' be quantified, such that it can be measured how easy it is to learn in an environment? with many agents?
- run large simulations over the weekend
  - ~~set variable path save location from batch file~~
  - ~~for shared exploration vs random exploration~~
  - occasionally remove one (or more) trained agents and replace them with "dumb" agents
  - ~~test with and without IoT nodes sharing historical information~~
    - ~~for without, zero out the inputs to the NNs~~ 