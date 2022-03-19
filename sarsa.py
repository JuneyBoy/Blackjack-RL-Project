from collections import defaultdict

import gym
import numpy as np

from plot_utils import *

env = gym.make('Blackjack-v1')
reward_dict = {'Losses': 0, 'Draws': 0, 'Wins': 0}
action_dict = {0: 'stick', 1: 'hit'}

def sarsa(env, num_episodes, alpha=0.05, gamma=0.5):
    '''
    Uses Sarsa (pg 130 in textbook)

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        alpha: step size or learning rate
        gamma: discount factor

    Returns:
        value function Q: mapping (state, action) -> value
    '''

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(num_episodes):
        state = env.reset()

        if state in Q:
            # if the state has been visited, then take the greedy action most of the time but take random action alpha % of the time
            action = np.random.choice([np.argmax(Q[state]),0,1], p=[1-alpha, 0.5*alpha, 0.5*alpha])
        else:
            # if the state has not been visited, then take random action
            action = np.random.choice([0,1])

        for i_step in range(100):
            
            next_state, reward, done, info = env.step(action)

            if next_state in Q:
                # if the state has been visited, then take the greedy action most of the time but take random action alpha % of the time
                next_action = np.random.choice([np.argmax(Q[next_state]),0,1], p=[1-alpha, 0.5*alpha, 0.5*alpha])
            else:
                # if the state has not been visited, then take random action
                next_action = np.random.choice([0,1])

            Q[state][action] = Q[state][action] + alpha*(reward + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action

            if done: break

    return Q


Q_Optimal = sarsa(env, 1000000)

pi = defaultdict(lambda: None)

for s in Q_Optimal:
    pi[s] = np.argmax(Q_Optimal[s])

env.close()
