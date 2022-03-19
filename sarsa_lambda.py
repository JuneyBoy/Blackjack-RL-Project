from collections import defaultdict

import gym
import numpy as np

from plot_utils import *

env = gym.make("Blackjack-v1")
reward_dict = {"Losses": 0, "Draws": 0, "Wins": 0}
action_dict = {0: "stick", 1: "hit"}


def under_17_policy(state):
    """
    Hit until player sum is 17 or greater.
    This is the same policy that the dealer uses.
    """
    sum, dealer, usable_ace = state
    return int(sum < 17)


def sarsa_lambda(env, num_episodes, gamma=0.5, alpha=0.05, slambda=0.5):
    """
    Sarsa(lambda) [slide 29 of Module 4 slides]

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        gamma: discount factor
        alpha: step size or learning rate
        slambda: trace-decay parameter

    Returns:
        value function Q: mapping (state, action) -> value
    """
    # map: state -> array of values (index: 0 -> value for stick, a=1 -> value for hit)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(num_episodes):
        e = defaultdict(lambda: np.zeros(env.action_space.n))
        state = env.reset()

        if state in Q:
            # if the state has been visited, then take the greedy action most of the time but take random action alpha % of the time
            action = np.random.choice(
                [np.argmax(Q[state])] + list(range(env.action_space.n)),
                p=[1 - alpha, 0.5 * alpha, 0.5 * alpha],
            )
        else:
            # if the state has not been visited, then take random action
            action = np.random.choice(list(range(env.action_space.n)))

        for i_step in range(100):
            next_state, reward, done, info = env.step(action)

            if next_state in Q:
                # if the state has been visited, then take the greedy action most of the time but take random action alpha % of the time
                next_action = np.random.choice(
                    [np.argmax(Q[next_state]), 0, 1],
                    p=[1 - alpha, 0.5 * alpha, 0.5 * alpha],
                )
            else:
                # if the state has not been visited, then take random action
                next_action = np.random.choice([0, 1])

            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            e[state][action] = e[state][action] + 1

            for s in Q.keys():
                for a in Q[s]:
                    Q[state][action] = (
                        Q[state][action] + alpha * delta * e[state][action]
                    )
                    e[state][action] = gamma * slambda * e[next_state][next_action]

            state = next_state
            action = next_action

            if done:
                break

    return Q


Q = sarsa_lambda(env, 400000)

pi = defaultdict(lambda: None)

for s in Q:
    pi[s] = np.argmax(Q[s])

plot_policy(pi)

env.close()
