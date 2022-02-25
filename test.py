from math import floor

import gym
import pygame
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


def under_17_policy(current_sum):
    '''
    Hit until player sum is 17 or greater.
    This is the same policy that the dealer uses.
    '''
    return int(current_sum < 17)


for i_episode in range(20):
    print()
    print("Episode {}:".format(i_episode))
    observation = env.reset()

    while True:
        env.render()  # draw the game
        pygame.event.pump()  # make game not crash on events

        print(observation)

        # get next action and observation from that action
        action = under_17_policy(observation[0])
        observation, reward, done, info = env.step(action)

        # wait for user input before taking next action
        input()

        print(action_dict[action])

        if done:
            break

    # final game observation and reward
    print(observation)
    print(reward)
    print()
    reward_dict[list(reward_dict)[floor(reward) + 1]] += 1

print("Wins: {}, Losses: {}, Draws: {}".format(reward_dict['Wins'],
                                               reward_dict['Losses'],
                                               reward_dict['Draws']))
env.close()
