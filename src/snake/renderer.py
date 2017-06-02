import time
import pygame as g

ELAPSED = 0.2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SMOKE = (233, 233, 233)
GRAY = (144, 144, 144)
PALETTE = [BLACK, WHITE, SMOKE, GRAY]
CELL = 20


class Renderer:
    def __init__(self, WIDTH, HEIGHT):
        self.width = WIDTH
        self.height = HEIGHT
        self.screen_size = (WIDTH * CELL, HEIGHT * CELL)
        self.screen = None
        self.clock = None

    def render(self, snake):
        if self.screen is None:
            g.init()
            self.screen = g.display.set_mode(self.screen_size)
            self.clock = g.time.Clock()
            g.display.set_caption('snake ai')
        states = snake.states
        width = snake.width
        height = snake.height
        self.screen.fill(BLACK)
        for x in range(width):
            for y in range(height):
                i = x + y * width
                state = states[i]
                if state is not 0:
                    rect = (x * CELL, y * CELL, CELL, CELL)
                    g.draw.rect(self.screen, PALETTE[state], rect, 0)
        g.display.update()
        # self.clock.tick()
        for e in g.event.get():
            if e.type is g.QUIT:
                return True
        return False
