"""
FOR PLOTTING BLACKJACK VALUE FUNCTIONS AND POLICIES
inspiration: https://github.com/udacity/deep-reinforcement-learning/blob/master/monte-carlo/plot_utils.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

action_dict = {0: 'stick', 1: 'hit'}


def print_policy(policy):
    for s in policy:
        print("policy({}): {}".format(s, action_dict[policy[s]]))


def print_Q(Q):
    for s in Q:
        print("Q({}) ... {}: {}, {}: {}".format(s, action_dict[0], Q[s][0],
                                                action_dict[1], Q[s][1]))


def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in V:
            return V[x, y, usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([
            get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))
        ]).reshape(X.shape)

        surf = ax.plot_surface(X,
                               Y,
                               Z,
                               rstride=1,
                               cstride=1,
                               cmap=plt.cm.coolwarm,
                               vmin=-1.0,
                               vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


def plot_policy(policy, title="Policy"):

    def get_Z(player_sum, dealer_show, usable_ace):
        if (player_sum, dealer_show, usable_ace) in policy:
            return policy[player_sum, dealer_show, usable_ace]
        else:
            return -1

    def get_figure(usable_ace, ax):
        x_range = np.arange(1, 11)
        y_range = np.arange(21, 10, -1)
        Z = np.array([[get_Z(y, x, usable_ace) for x in x_range]
                      for y in y_range])
        surf = ax.imshow(Z,
                         cmap=plt.get_cmap(
                             ListedColormap(["#000000", "#FF5A5A", "#9BFF80"]),
                             3),
                         vmin=-1,
                         vmax=1,
                         extent=[0.5, 10.5, 10.5, 21.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        ax.set_xlabel('Dealer\'s Showing Card')
        ax.set_ylabel('Player\'s Current Sum')
        ax.grid(color='w', linestyle='-', linewidth=0.169)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[-1, 0, 1], cax=cax)
        cbar.ax.set_yticklabels(['-1 (DNE)', '0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(7, 7))
    plt.suptitle(title, fontsize=13)
    ax = fig.add_subplot(211)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.tight_layout()
    plt.show()
