from collections import defaultdict

import gym
import numpy as np

from plot_utils import *

env = gym.make('Blackjack-v1')
reward_dict = {'Losses': 0, 'Draws': 0, 'Wins': 0}
action_dict = {0: 'stick', 1: 'hit'}


def under_17_policy(state):
    '''
    Hit until player sum is 17 or greater.
    This is the same policy that the dealer uses.
    '''
    sum, dealer, usable_ace = state
    return int(sum < 17)

def mc_es(policy, env, num_episodes, gamma=1):
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
    # map: state -> action
    pi = defaultdict(lambda: None)

    # map: state -> array of values (index: 0 -> value for stick, a=1 -> value for hit)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # for each state-action pair: track return value sum, and number of times it occurs
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
        G = 0
        for (s, a, r) in episode:
            G = gamma * G + r
            # each step of episode is unique
            return_sum[s][a] += G
            return_count[s][a] += 1.0
            Q[s][a] = return_sum[s][a] / return_count[s][a]
            pi[s] = np.argmax(Q[s])

    return Q, pi


Q, pi = mc_es(under_17_policy, env, 100000)

env.close()
