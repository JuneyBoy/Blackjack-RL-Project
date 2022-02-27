from collections import defaultdict

import gym
import numpy as np

from plot_utils import *

env = gym.make('Blackjack-v1')
reward_dict = {'Losses': 0, 'Draws': 0, 'Wins': 0}
action_dict = {0: 'stick', 1: 'hit'}


def print_policy(policy):
    for s in policy:
        print("policy({}): {}".format(s, action_dict[policy[s]]))


def print_value(Q):
    for s in Q:
        print("Q({}) ... {}: {}, {}: {}".format(s, action_dict[0], Q[s][0],
                                                action_dict[1], Q[s][1]))


def under_17_policy(state):
    '''
    Hit until player sum is 17 or greater.
    This is the same policy that the dealer uses.
    '''
    sum, dealer, usable_ace = state
    return int(sum < 17)


def under_20_policy(state):
    '''
    Hit until player sum is 20 or greater. Sticks only on 20 or 21
    '''
    sum, dealer, usable_ace = state
    return int(sum < 20)


def mc_es(policy, env, num_episodes, gamma=1.0):
    '''
    Uses monte carlo exploring starts (ES) prediction (pg 99 in textbook)

    Args:
        policy: arbitrary policy to be used in MC algorithm, mapping state -> action.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        gamma: Gamma value in MC algorithm.

    Returns:
        value function Q: mapping (state, action) -> value
        optimized policy pi: mapping state -> action
    '''
    pi = defaultdict(lambda: None)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    return_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    return_count = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(num_episodes):
        # generate episode
        episode = []
        state = env.reset()
        while True:
            # get action based on pi (if exists) or given policy
            action = pi[state] if pi[state] != None else policy(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))

            # next state
            state = next_state
            if done: break

        # calculations
        for s, a, r in episode:
            # each step of episode is unique
            # use reward from the end of episode
            return_sum[s][a] += r
            return_count[s][a] += 1.0
            Q[s][a] = return_sum[s][a] / return_count[s][a]
            pi[s] = np.argmax(Q[s])

    return Q, pi


Q17, pi17 = mc_es(under_17_policy, env, 100000)
# Q20, pi20 = mc_es(under_20_policy, env, 100000)

# plot_policy(pi17, "under_17_policy")
# plot_policy(pi20, "under_20_policy")

env.close()
