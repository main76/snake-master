from random import randint
import numpy

START_LENGTH = 6

REWARD = 1
NO_REWARD = 0
PUNISHMENT = -0.5


class Snake:
    def __init__(self, shape):
        width, height = shape
        area = width * height
        head = int(height / 2) * width + int((width - START_LENGTH) / 2)
        self.body = list(range(head, head + START_LENGTH))
        self.shape = shape
        self.area = area
        self.width = width
        self.height = height
        self.moves = []
        self.scores = 0
        self.food = self.__refresh_food()
        self.__moves = [self.__up, self.__left, self.__down, self.__right]

    def move(self, action):
        if abs(action) > 2:
            raise 'Argument exception!'
        # action:
        # 0 - turn clockwisely
        # 1 - do not turn
        # 2 - turn counterclockwisely
        direction = (action - 1 + self.heading) % 4  # magic
        move = self.__moves[direction]
        self.moves.append(action)
        return move()

    def __refresh_food(self):
        return randint(0, self.spaces - 1)

    def __left(self):
        head = self.head
        if head % self.width == 0:
            return PUNISHMENT, True
        return self.__move(head - 1)

    def __right(self):
        head = self.head
        if head % self.width == self.width - 1:
            return PUNISHMENT, True
        return self.__move(head + 1)

    def __up(self):
        head = self.head
        if head < self.width:
            return PUNISHMENT, True
        return self.__move(head - self.width)

    def __down(self):
        head = self.head
        if head >= self.area - self.width:
            return PUNISHMENT, True
        return self.__move(head + self.width)

    def __move(self, p):
        if self.food == p:
            self.scores += 1
            self.body.insert(0, p)
            done = len(self.body) == self.spaces
            if not done:
                self.__refresh_food()
            return REWARD, done
        self.body.pop()
        if p in self.body:
            return PUNISHMENT, True
        self.body.insert(0, p)
        return NO_REWARD, False

    @property
    def heading(self):
        head = self.head
        headnext = self.headnext
        if head == headnext - self.width:
            return 0  # w
        elif head == headnext - 1:
            return 1  # a
        elif head == headnext + self.width:
            return 2  # s
        elif head == headnext + 1:
            return 3  # d
        else:
            raise 'Ooops!'

    @property
    def spaces(self):
        return self.area - len(self.body)

    @property
    def head(self):
        return self.body[0]

    @property
    def headnext(self):
        return self.body[1]

    @property
    def states(self):
        states = numpy.zeros(self.area, dtype=int)
        for node in self.body:
            states[node] = BODY
        states[self.head] = HEAD
        index = 0
        for i in range(self.area):
            if index == self.food:
                states[i] = FOOD
                break
            elif states[i] is SPACE:
                index += 1
        return states


# ENUM
SPACE = 0
BODY = 1
HEAD = 2
FOOD = 3
