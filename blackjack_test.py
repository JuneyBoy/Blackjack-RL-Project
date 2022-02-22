from collections import defaultdict, namedtuple
from pickle import FALSE

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards'])


def plot_value_function(V, title="Value Function"):
    '''
    Plots the value function as a surface plot.
    '''
    min_x = 11  # min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2,
                                  np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2,
                                np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X,
                               Y,
                               Z,
                               rstride=1,
                               cstride=1,
                               cmap=matplotlib.cm.coolwarm,
                               vmin=-1.0,
                               vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


env = gym.make('Blackjack-v1')  # insert your favorite environment
env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
for _ in range(100):
    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = env.action_space.sample()
    env.step(action)


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    '''
    Monte Carlo prediction algorithm. 
    Calculates the value function for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    '''
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for i_episode in range(num_episodes):
        observation = env.reset()

        episodes = []
        for i in range(100):
            action = policy(observation)
            next_observation, reward, done, _ = env.step(action)
            episodes.append((observation, action, reward))
            if done:
                break
            observation = next_observation

        # obtain unique observation set
        observations = set([x[0] for x in episodes])
        for i, observation in enumerate(observations):
            # first occurence of the observation
            idx = episodes.index([
                episode for episode in episodes if episode[0] == observation
            ][0])

            Q = sum([
                episode[2] * discount_factor**i for episode in episodes[idx:]
            ])

            returns_sum[observation] += Q
            returns_count[observation] += 1.0

            V[observation] = returns_sum[observation] / \
                returns_count[observation]

    return V


#env = gym.make('Blackjack-v1', FALSE)


# A policy that sticks if the player score is >= 20 and hits otherwise.
def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plot_value_function(V_10k, title='10000 Steps')

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plot_value_function(V_500k, title='500000 Steps')
