from .snake import Snake
from .renderer import Renderer


class Handler:
    def __init__(self, shape, onexit=None):
        self.shape = shape
        self.renderer = Renderer(shape)
        self.onexit = onexit
        self.steps = 0
        self.reset()

    def reset(self):
        self.snake = Snake(self.shape)
        return self.snake.states

    def step(self, action):
        reward, done = self.snake.move(action)
        info = None if not done else 'moves: %d, scores: %d' % (
            self.snake.total_moves, self.snake.scores)
        return self.snake.states, reward, done, info

    def render(self):
        close = self.renderer.render(self.snake)
        if close:
            if self.onexit is not None:
                self.onexit()
            exit(0)

    @property
    def moves(self):
        return self.snake.moves
