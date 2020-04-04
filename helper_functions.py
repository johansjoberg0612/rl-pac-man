import numpy as np

def prepro(I):
    """
    preprocess 210x160x3 uint8 frame into 6400 (100x80) 1D float vector
    """
    I = I[:200]  # crop so that we don't use bottom rows
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r, gamma):
    """
    take 1D float array of rewards and compute discounted reward where the further away a move was from the reward the
    less the move is persumed to have contributed to it
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Gets the values of the outputs neurons and performs softmax activation
def get_prob_multiclass(model, h):
    p = np.zeros((model['W2'].shape[1]))
    for i in range(model['W2'].shape[1]):
        logp = np.dot(model['W2'][:, i], h)
        p[i] = np.exp(logp)
    p /= sum(p)
    return p

def policy_forward(x, model):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    p = get_prob_multiclass(model, h)
    return p, h  # returns the probability of taking each action as an array, and hidden state

def policy_backward(eph, epdlogp, epx, model):
    """ Gives the gradients of the filter weights, eph is a vector of hidden states,
    epdlogp is losses and epx is the inputs to the forward pass"""

    dW2 = np.zeros((200, 5))
    for i in range(5):
        dW2[:, i] = np.dot(eph.T, epdlogp[:, i]).ravel()
    dh = np.dot(epdlogp, model['W2'].T)
    dh[eph <= 0] = 0  # backprop relu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def get_difference(action, aprob):
    # Creating getting one-hot representatation of the action taken
    y = np.zeros((5))
    y[(action)] = 1
    return (y - aprob)


class StateStorer:
    def __init__(self):
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []

    def append_states(self, x, h, dlogp, r):
        self.xs.append(x)  # observation
        self.hs.append(h)  # hidden state
        self.dlogps.append(dlogp)
        self.drs.append(r)  # record reward (has to be done after we call step() to get reward for previous action)

    def get_states(self):
        epx = np.vstack(self.xs)
        eph = np.vstack(self.hs)
        epdlogp = np.vstack(self.dlogps)
        epr = np.vstack(self.drs)
        return epx, eph, epdlogp, epr

    def reset_states(self):
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []
