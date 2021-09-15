from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

import random
import numpy as np
import pandas as pd
import json

class Q_learning():
    def __init__(self, num_actions, load_Q_table=None, alpha=0.3, epsilon=0.8, gamma= 0.95):
        """
        num_actions - integer. How many actions agent can do
        load_Q_table - string. Name of JSON file to load Q table from
        alpha - float. discount factor for calculation of Q value 
        epsilon - float. Probability threshold do be greedy or not
        gamma - float. discount factor for calculation of Q value 

        """
        if load_Q_table is None:
            self.Q_table = {}
        else:
            self.Q_table = json.load( open(load_Q_table ) )
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_action(self, state, action_mask): # In action_mask valule 0 - available action, 1 - not available 
        if state not in self.Q_table.keys():
            self.Q_table[state] = [0.] * self.num_actions # add state to Q table if it doesn't exist

        Chosen_action = None
        # epsilon greedy action choice. Action needs to be numpy array with 1 value. 0, 1 or 2
        # ---------------------------
        if random.random() > self.epsilon:
            Chosen_action = np.array(self.Q_table[state].index(max(self.Q_table[state] ) ) ) # greedy
        else:  # random action
            while Chosen_action == None:
                suggested_action_idx = random.choice(range(self.num_actions ) )
                if action_mask[suggested_action_idx] == 0:
                    Chosen_action = np.array(suggested_action_idx)
        # ---------------------------
        Chosen_action.resize((1, 1))
        return Chosen_action

    def update_Q_table(self, state, action, next_state, reward, next_action=None, finished=False):
        if state not in self.Q_table.keys():
            self.Q_table[state] = [0.] * self.num_actions # add state to Q table if it doesn't exist
        if next_state not in self.Q_table.keys():
            self.Q_table[next_state] = [0.] * self.num_actions # add next_state to Q table if it doesn't exist
        # Updating Q value
        # ---------------------------
        if finished:
            self.Q_table[state][action] += self.alpha *  (reward - self.Q_table[state][action] )
        else:
            self.Q_table[state][action] += self.alpha * ( reward + self.gamma * max(self.Q_table[next_state] )   - self.Q_table[state][action] )
        # ---------------------------

    def save_Q_table(self, filename="Q_table.json"):
        json.dump( self.Q_table, open( filename, 'w' ) )

    def getQ_table(self):
        return pd.DataFrame.from_dict(self.Q_table, orient='index',
                       columns=[f"Action_{x}" for x in range(self.num_actions )]).sort_index()

if __name__ == "__main__":

    # This is a non-blocking call that only loads the environment.
    print ("Script started. Please start Unity environment to start training proccess.")
    env = UnityEnvironment( seed=1, side_channels=[])
    # Start interacting with the environment.
    env.reset()
    # Info about our environment ---------------------
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    action_spec = spec.action_spec
    print (behavior_name)
    print (spec)
    # Examine the number of observations per Agent
    print("Number of observations : ", len(spec.observation_shapes))
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
    for index, shape in enumerate(spec.observation_shapes):
        if len(shape) == 1:
            print("First vector observations : ", decision_steps.obs[index][0,:])
    # Info about our environment ---------------------

    NUM_EPISODES = 100 # number of episodes for training proccess

    # create our brain
    agent = Q_learning(num_actions=action_spec.discrete_branches[0], alpha=0.35, epsilon=0.85, gamma= 0.95 )

    for episode in range(NUM_EPISODES):
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

            # Generate an action for all agents
            state = int(np.argmax( decision_steps[tracked_agent].obs[0] ) )
            actions_discrete = agent.choose_action(state, decision_steps[tracked_agent].action_mask[0])
            actions = ActionTuple(discrete=actions_discrete)
            # Set the actions
            env.set_actions(behavior_name, actions)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if tracked_agent in decision_steps: # The agent requested a decision
                new_state = int(np.argmax( decision_steps[tracked_agent].obs[0] ) )
                agent.update_Q_table(state, actions_discrete[0][0], new_state, decision_steps[tracked_agent].reward)
                episode_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps: # The agent terminated its episode
                new_state = int(np.argmax( terminal_steps[tracked_agent].obs[0] ) )
                agent.update_Q_table(state, actions_discrete[0][0], new_state, terminal_steps[tracked_agent].reward, finished=True)
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True

        print(f"Total rewards for episode {episode} is {episode_rewards}")
        if episode % 100 == 0:
            print (f"Current Q table:\n {agent.getQ_table()}")
            agent.save_Q_table()
    print (f"Current Q table:\n {agent.getQ_table()}")
    print ("Finished training!!! \nSaving Q table into file")
    agent.save_Q_table()
    env.close()