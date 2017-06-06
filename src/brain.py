import cntk as C
import numpy as np

LEARNING_RATE = 0.2

HIDDEN_DIM = 512


class Brain:
    def __init__(self, action_count, input_shape=None, model_path=None):
        if model_path is None and input_shape is None:
            raise 'Ooops!'
        if model_path is None:
            self.model, self.trainer, self.loss = self.__create(input_shape, action_count)
        else:
            self.model, self.trainer, self.loss = self.__load(model_path, action_count)

    # Correspoding layers implementation - Preferred solution
    def create_model(self, input, action_count):
        z = C.layers.Sequential([
                C.layers.Convolution2D((3, 3), 16, activation=C.ops.relu),
                C.layers.MaxPooling((2, 2), (2, 2)),
                C.layers.Convolution2D((3, 3), 32, activation=C.ops.relu),
                C.layers.MaxPooling((2, 2), (2, 2)),
                C.layers.Dense(action_count)
            ])
        return z(input)

    def __load(self, model_path, action_count):
        model = C.load_model(model_path)        
        trainer, loss = self.__get_trainer_loss(model, action_count)
        return model, trainer, loss

    def __create(self, input_shape, action_count):
        observation = C.sequence.input_variable(input_shape, np.float32)
        model = self.create_model(observation, action_count)
        trainer, loss = self.__get_trainer_loss(model, action_count)
        return model, trainer, loss

    def __get_trainer_loss(self, model, action_count):
        q_target = C.sequence.input_variable(action_count, np.float32)

        # loss='mse'
        loss = C.reduce_mean(C.square(model - q_target), axis=0)
        meas = C.reduce_mean(C.square(model - q_target), axis=0)

        # optimizer
        lr_schedule = C.learning_rate_schedule(LEARNING_RATE, C.UnitType.minibatch)
        learner = C.sgd(
            model.parameters,
            lr_schedule,
            gradient_clipping_threshold_per_sample=10)
        trainer = C.Trainer(model, (loss, meas), learner)

        return trainer, loss

    def train(self, x, y):
        arguments = dict(zip(self.loss.arguments, [x, y]))
        updated, results = self.trainer.train_minibatch(
            arguments, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval([s])
