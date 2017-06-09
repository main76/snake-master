import math
import random
import numpy as np
from brain import Brain
from memory import Memory

MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

GAMMA = 0.8  # discount factor

MAX_EPSILON = 1
MIN_EPSILON = 0.01  # stay a bit curious even when getting old
LAMBDA = 0.00001  # speed of decay


class Agent:
    def __init__(self, input_shape, action_count, steps=0, model_path=None):
        self.steps = steps
        self.epsilon = MAX_EPSILON if steps == 0 else self.__calc_epsilon(steps)
        self.brain = Brain(action_count, input_shape=input_shape, model_path=model_path)
        self.memory = Memory(MEMORY_CAPACITY)
        self.input_shape = input_shape
        self.action_count = action_count

    def act(self, s):
        action = -1
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_count - 1)
        else:
            predictions = np.squeeze(self.brain.predict(s.astype(np.float32)))
            action = round(np.argmax(predictions))
            weight_sqrsum = 0
            for i in range(self.action_count):
                if predictions[i] < 0 or predictions[i] * 2 < predictions[action]:
                    predictions[i] = 0
                else:
                    weight_sqrsum += math.pow(predictions[i], 2)
            if weight_sqrsum != 0:
                dice = random.random() * weight_sqrsum
                weight_begin = 0
                for i in range(self.action_count):
                    if weight_begin < dice and dice < weight_begin + math.pow(predictions[i], 2):
                        action = i
                        break
                    else:
                        weight_begin = math.pow(predictions[i], 2)                
        return action

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def __calc_epsilon(self, steps):
        return MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

    def replay(self, batch_size=BATCH_SIZE):
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = self.__calc_epsilon(self.steps)

        batch = self.memory.sample(batch_size)
        batch_len = len(batch)

        no_state = np.zeros(self.input_shape)

        # CNTK: explicitly setting to float32
        states = np.array([o[0] for o in batch], dtype=np.float32)
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch], dtype=np.float32)

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        # CNTK: explicitly setting to float32
        x = np.zeros((batch_len, *self.input_shape)).astype(np.float32)
        y = np.zeros((batch_len, self.action_count)).astype(np.float32)

        for i in range(batch_len):
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
