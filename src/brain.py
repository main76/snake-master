import cntk as C
import numpy as np

HIDDEN_DIM = 512  # hidden layer size


class Brain:
    def __init__(self, STATE_COUNT, ACTION_COUNT):
        self.model, self.trainer, self.loss = self.__create(
            STATE_COUNT, ACTION_COUNT)

    # Correspoding layers implementation - Preferred solution
    def create_model(self, input, ACTION_COUNT):
        with C.layers.default_options(init=C.glorot_uniform()):
            z = C.layers.Sequential([
                C.layers.Dense(HIDDEN_DIM, name="l1"),
                C.layers.Dense(ACTION_COUNT, name="l2")
            ])
            return z(input)

    def __create(self, STATE_COUNT, ACTION_COUNT):
        observation = C.sequence.input_variable(
            STATE_COUNT, np.float32, name="s")
        q_target = C.sequence.input_variable(
            ACTION_COUNT, np.float32, name="q")

        model = model = self.create_model(observation, ACTION_COUNT)

        # loss='mse'
        loss = C.reduce_mean(C.square(model - q_target), axis=0)
        meas = C.reduce_mean(C.square(model - q_target), axis=0)

        # optimizer
        lr = 0.00025
        lr_schedule = C.learning_rate_schedule(lr, C.UnitType.minibatch)
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
