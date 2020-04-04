"""
Main script for model training
"""
import numpy as np
import pickle
import gym
from helper_functions import prepro, policy_forward, policy_backward, get_difference, discount_rewards, StateStorer

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 3  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.90  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = True  # resume from previous checkpoint?
render = True #Whether to render the pacman game

# model initialization
D = 100 * 80  # input dimensionality: 100x80 grid
if resume:
    model = pickle.load(open('pacman.pickle', 'rb'))
else:
    """
    Defines a simple neural network with one hidden layer and H neurons
    """
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    # Action space of pacman is 5 and the out put probs will be for ['NOOP','UP', 'RIGHT', 'LEFT', 'DOWN']
    model['W2'] = np.random.randn(H, 5) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


env = gym.make("MsPacman-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame

#Object for storing states, hidden layers, losses and rewards for each episode
States=StateStorer()

running_reward = None #Rolling average of rewards over 100 episodes
reward_sum = 0 #Total reward for the episode
episode_number = 0

# To keep track of whether a ghost have hit us
current_number_of_lives = 3
prev_number_of_lives = 3
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x, model)
    action = np.random.choice([0, 1, 2, 3, 4], p=aprob)  # roll the dice!

    logp=get_difference(action, aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)

    # Penalising being killed by a ghost
    prev_number_of_lives = current_number_of_lives
    current_number_of_lives = info['ale.lives']
    if current_number_of_lives < prev_number_of_lives:
        reward -= 500

    reward_sum += reward

    States.append_states(x, h, logp, reward)
    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode

        epx, eph, epdlogp, epr= States.get_states()

        States.reset_states()

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr, gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp, epx, model)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                #Update rms_propcache
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 30 == 0: pickle.dump(model, open('pacman.pickle', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
    #Print reward
    if reward != 0:
        print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
