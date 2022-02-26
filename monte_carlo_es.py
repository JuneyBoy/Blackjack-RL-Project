from collections import defaultdict

import gym
import numpy as np
"""
Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.
    
    ### Description
    Card Values:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.
    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.
    
    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.
    
    ### Action Space
    There are two actions: stick (0), and hit (1).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).
    
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).
    
    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
"""
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
        policy: A function that maps an state to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        gamma: Gamma value in monte carlo algorithm.

    Returns:
        value function V: mapping state to real numbers
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
            return_sum[s][a] += reward
            return_count[s][a] += 1.0
            Q[s][a] = return_sum[s][a] / return_count[s][a]
            pi[s] = np.argmax(Q[s])

    return Q, pi


Q, pi = mc_es(under_17_policy, env, 50000, 0.69)

for s in Q:
    print("Q({}) ... {}: {}, {}: {}".format(s, action_dict[0], Q[s][0],
                                            action_dict[1], Q[s][1]))

print()

for s in pi:
    print("pi({}): {}".format(s,
                              action_dict[pi[s]] if pi[s] != None else pi[s]))

env.close()
