from .snake import Snake, CHANNEL
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
        close = self.renderer.render(self.snake.states)
        if close:
            if self.onexit is not None:
                self.onexit()
            exit(0)

    def screenshot(self, states, output_path):
        self.renderer.screenshot(states, output_path)

    @property
    def moves(self):
        return self.snake.paces
