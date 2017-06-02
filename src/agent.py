import math
import random
import numpy as np
from brain import Brain
from memory import Memory

MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

GAMMA = 0.90  # discount factor

MAX_EPSILON = 1
MIN_EPSILON = 0.01  # stay a bit curious even when getting old
LAMBDA = 0.0001  # speed of decay


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, STATE_COUNT, ACTION_COUNT):
        self.brain = Brain(STATE_COUNT, ACTION_COUNT)
        self.memory = Memory(MEMORY_CAPACITY)
        self.state_count = STATE_COUNT
        self.action_count = ACTION_COUNT

    def act(self, s):
        action = 0
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_count - 1)
        else:
            predictions = self.brain.predict(s.astype(np.float32))
            action = round(np.argmax(predictions))
        return action

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (
            MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self, batch_size = BATCH_SIZE):
        batch = self.memory.sample(batch_size)
        batchLen = len(batch)

        no_state = np.zeros(self.state_count)

        # CNTK: explicitly setting to float32
        states = np.array([o[0] for o in batch], dtype=np.float32)
        states_ = np.array(
            [(no_state if o[3] is None else o[3]) for o in batch],
            dtype=np.float32)

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        # CNTK: explicitly setting to float32
        x = np.zeros((batchLen, self.state_count)).astype(np.float32)
        y = np.zeros((batchLen, self.action_count)).astype(np.float32)

        for i in range(batchLen):
            s, a, r, s_ = batch[i]

            # CNTK: [0] because of sequence dimension
            t = p[0][i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[0][i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
