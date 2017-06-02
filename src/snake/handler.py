from .snake import Snake
from .renderer import Renderer


class Handler:
    def __init__(self, width, height, onexit):
        self.shape = (width, height)
        self.renderer = Renderer(width, height)
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
            self.onexit()
            exit(0)

    @property
    def moves(self):
        return len(self.snake.moves)
