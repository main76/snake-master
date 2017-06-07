from .snake import EXIST

ELAPSED = 0.2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SMOKE = (233, 233, 233)
GRAY = (144, 144, 144)
PALETTE = [WHITE, SMOKE, GRAY]
CELL = 20


class Renderer:
    def __init__(self, shape):
        width, height = shape
        self.width = width
        self.height = height
        self.screen_size = (width * CELL, height * CELL)
        self.screen = None
        self.clock = None

    def render(self, snake):
        import pygame as g
        if self.screen is None:
            g.init()
            self.screen = g.display.set_mode(self.screen_size)
            self.clock = g.time.Clock()
            g.display.set_caption('snake ai')
        states = snake.states
        channel, width, height = states.shape
        self.screen.fill(BLACK)
        for c in range(channel):
            for x in range(width):
                for y in range(height):
                    state = int(states[c][y][x])
                    if state is EXIST:
                        rect = (x * CELL, y * CELL, CELL, CELL)
                        g.draw.rect(self.screen, PALETTE[c], rect, 0)
        g.display.update()
        # self.clock.tick()
        for e in g.event.get():
            if e.type is g.QUIT:
                return True
        return False
