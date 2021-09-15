from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

import numpy as np
import json
import sys

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x



if __name__ == "__main__":

    # Load learnt Q table

    # input file name from command line. 
    # exmaple: python inference_Basic.py Q_table.json
    # exmaple: python inference_Basic.py Q_table_SARSA.json
    Q_table = json.load( open(sys.argv[1]), object_hook=jsonKeys2int )

    # channel = EngineConfigurationChannel()
    # channel.set_configuration_parameters(time_scale = 1, target_frame_rate = 1) # sets how fast simulation runs
    # This is a non-blocking call that only loads the environment.
    print ("Script started. Please start Unity environment to start training proccess.")
    env = UnityEnvironment( seed=1, side_channels=[])
    
    # Start interacting with the environment.
    env.reset()
    
    # print (dir(env))
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    action_spec = spec.action_spec
    print (behavior_name)
    print (spec)
    # Examine the number of observations per Agent
    print("Number of observations : ", len(spec.observation_specs))

    # Is the Action continuous or multi-discrete ?
    if action_spec.is_continuous():
        print("The action is continuous")
    # print (spec.action_spec.random_action() )
    if action_spec.is_discrete():
        print("The action is discrete")

    # How many actions are possible ?
    print(f"There are {action_spec.discrete_size} action(s)")

    if action_spec.is_discrete():
        for action, branch_size in enumerate(action_spec.discrete_branches):
            print(f"Action number {action} has {branch_size} different options")

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    for index, shape in enumerate(spec.observation_specs):
        if len(shape) == 1:
            print("First vector observations : ", decision_steps.obs[index][0,:])

    NUM_EPISODES = 10 # number of episodes for inference proccess

    

    for episode in range(NUM_EPISODES):
        # for ste in range(NUM_STEPS):
        env.reset() # reset environment on each episode
        # Get info about starting agents states
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent
        episode_rewards = 0 # For the tracked_agent
        # Play in an environment until reach goal or terminated
        while not done:
            
            # Track the first agent we see if not tracking 
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0] 


            state = int(np.argmax( decision_steps[tracked_agent].obs[0] ) )
            actions_discrete = np.array(Q_table[state].index(max(Q_table[state] ) ) ) # choose action with max Q value in that state
            actions_discrete.resize((1, 1))
            print (f"Action chosen: {actions_discrete}")
            actions = ActionTuple(discrete=actions_discrete)
            # print (f"action: {actions.discrete}")
            # Set the actions
            env.set_actions(behavior_name, actions)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if tracked_agent in decision_steps: # The agent requested a decision
            #     new_state = int(np.argmax( decision_steps[tracked_agent].obs[0] ) )
            #     agent.update_Q_table(state, actions_discrete[0][0], new_state, decision_steps[tracked_agent].reward)
                episode_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps: # The agent terminated its episode
            #     new_state = int(np.argmax( terminal_steps[tracked_agent].obs[0] ) )
            #     agent.update_Q_table(state, actions_discrete[0][0], new_state, terminal_steps[tracked_agent].reward)
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True

        print(f"Total rewards for episode {episode} is {episode_rewards}")         

    env.close()