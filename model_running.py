"""
Script for running model
"""
import numpy as np
import pickle
import gym
from helper_functions import prepro, policy_forward

#Load model from file
model = pickle.load(open('pacman.pickle', 'rb'))

D = 100 * 80  # input dimensionality: 100x80 grid


env = gym.make("MsPacman-v0")
observation = env.reset()

prev_x = None  # used in computing the difference frame

running_reward = None #Rolling average of rewards over 100 episodes
reward_sum = 0 #Total reward for the episode
episode_number = 0


while True:
    env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x, model)
    action = np.random.choice([0, 1, 2, 3, 4], p=aprob)  # roll the dice!

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)

    reward_sum += reward

    if done:  # an episode finished
        episode_number += 1

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 30 == 0: pickle.dump(model, open('pacman.pickle', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
