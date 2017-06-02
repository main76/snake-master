import cntk as C
import numpy as np

LEARNING_RATE = 0.001

HIDDEN_DIM = 256
LSTM_DIM = 128

class Brain:
    def __init__(self, state_count, action_count):
        self.model, self.trainer, self.loss = self.__create(
            state_count, action_count)

    # Correspoding layers implementation - Preferred solution
    def create_model(self, input, action_count):
        z = C.layers.Sequential([
            C.layers.Dense(HIDDEN_DIM),
            C.layers.Recurrence(C.layers.LSTM(LSTM_DIM)),
            C.layers.Dense(action_count)
        ])
        return z(input)

    def __create(self, state_count, action_count):
        observation = C.sequence.input_variable(state_count, np.float32)
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

    def train(self, x, y, epoch=1, verbose=0):
        #self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)
        arguments = dict(zip(self.loss.arguments, [x, y]))
        updated, results = self.trainer.train_minibatch(
            arguments, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval([s])
