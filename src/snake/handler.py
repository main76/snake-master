from .snake import Snake
from .renderer import Renderer


class Handler:
    def __init__(self, shape, onexit = None):
        self.shape = shape
        self.renderer = Renderer(shape)
        self.onexit = onexit
        self.reset()

    def reset(self):
        self.snake = Snake(self.shape)
        return self.snake.states

    def step(self, action):
        reward, done = self.snake.move(action)
        info = None if not done else 'moves: %d, scores: %d, actions: [ %s ]' % (
            self.moves, self.snake.scores,
            ', '.join(str(x) for x in self.snake.moves))
        return self.snake.states, reward, done, info

    def render(self):
        close = self.renderer.render(self.snake)
        if close:
            if self.onexit is not None:
                self.onexit()
            exit(0)

    @property
    def moves(self):
        return len(self.snake.moves)
