from collections import defaultdict

import gym
import numpy as np

import monte_carlo_es as mces
import q_learning as ql
import sarsa as sarsa
from plot_utils import *

env = gym.make('Blackjack-v1')

def play_blackjack(env, num_episodes, policy=None):

    reward_dict = {-1: 0, 0: 0, 1: 0}

    for i_episode in range(num_episodes):
        state = env.reset()

        for i_step in range(100):
            
            if policy != None:
                action = policy[state]
            else:
                action = state[0] < 17

            state, reward, done, info = env.step(action)

            if done:
                # records result of episode
                reward_dict[int(np.floor(reward))] += 1
                break
    
    return reward_dict


under_17_results = play_blackjack(env, 100000)
mces_results = play_blackjack(env, 100000, mces.pi)
ql_results = play_blackjack(env, 100000, ql.pi)
sarsa_results = play_blackjack(env, 100000, sarsa.pi)

print("Under 17 Results: {} Losses      {} Draws        {} Wins".format(under_17_results[-1], under_17_results[0], under_17_results[1]))
print("Monte Carlo ES Rsults: {} Losses      {} Draws        {} Wins".format(mces_results[-1], mces_results[0], mces_results[1]))
print("Q Learning Results: {} Losses      {} Draws        {} Wins".format(ql_results[-1], ql_results[0], ql_results[1]))
print("Sarsa Learning Results: {} Losses      {} Draws        {} Wins".format(sarsa_results[-1], sarsa_results[0], sarsa_results[1]))

plot_policy(mces.pi, r'Monte Carlo ES - $\pi_*$')
plot_policy(ql.pi, r'Q Learning - $\pi_*$')
plot_policy(sarsa.pi, r'Sarsa - $\pi_*$')

env.close()
