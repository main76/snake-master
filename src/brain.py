import cntk as C
import numpy as np

LEARNING_RATE = 0.2

HIDDEN_DIM = 512


class Brain:
    def __init__(self, input_shape, action_count):
        self.model, self.trainer, self.loss = self.__create(
            input_shape, action_count)

    # Correspoding layers implementation - Preferred solution
    def create_model(self, input, action_count):
        # with C.layers.default_options(init=C.glorot_uniform()):
        #     z = C.layers.Sequential(
        #         [C.layers.Dense(HIDDEN_DIM),
        #          C.layers.Dense(action_count)])
        #     return z(input)

        c1 = C.layers.Convolution2D((3, 3), 16, activation=C.ops.relu)(input)
        m1 = C.layers.MaxPooling((2, 2), (2, 2))(c1)
        c2 = C.layers.Convolution2D((3, 3), 32, activation=C.ops.relu)(m1)
        m2 = C.layers.MaxPooling((2, 2), (2, 2))(c2)
        z = C.layers.Dense(action_count)(m2)
        return z

    def __create(self, input_shape, action_count):
        observation = C.sequence.input_variable(input_shape, np.float32)
        q_target = C.sequence.input_variable(action_count, np.float32)

        model = model = self.create_model(observation, action_count)

        # loss='mse'
        loss = C.reduce_mean(C.square(model - q_target), axis=0)
        meas = C.reduce_mean(C.square(model - q_target), axis=0)

        # optimizer
        lr_schedule = C.learning_rate_schedule(LEARNING_RATE,
                                               C.UnitType.minibatch)
        learner = C.sgd(
            model.parameters,
            lr_schedule,
            gradient_clipping_threshold_per_sample=10)
        trainer = C.Trainer(model, (loss, meas), learner)

        # CNTK: return trainer and loss as well
        return model, trainer, loss

    def train(self, x, y):
        arguments = dict(zip(self.loss.arguments, [x, y]))
        updated, results = self.trainer.train_minibatch(
            arguments, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval([s])
