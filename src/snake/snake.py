from random import randint
import numpy

START_LENGTH = 6

REWARD = 1
NO_REWARD = -0.005
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
        self.total_moves = 0
        self.start = 0
        self.scores = 0
        self.food = self.__refresh_food()
        self.paces = 0
        self.__states = None
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
        self.total_moves += 1
        self.__states = None
        return move()

    def __refresh_food(self, body_state=None):
        if body_state is None:
            body_state = numpy.zeros(self.area, dtype=int)
            for node in self.body:
                body_state[node] = EXIST
        else:
            body_state = numpy.reshape(body_state, self.area)
        ti = randint(0, self.spaces - 1)
        index = 0
        for i in range(self.area):
            if index == ti:
                return i
            elif int(body_state[i]) is not EXIST:
                index += 1
        raise 'Ooops!'

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
            self.scores += REWARD
            self.paces = self.total_moves - self.start
            self.start = self.total_moves
            self.body.insert(0, p)
            done = len(self.body) == self.spaces
            if not done:
                self.food = self.__refresh_food(self.states[BODY])
            return REWARD, done
        self.body.pop()
        if p in self.body:
            return PUNISHMENT, True
        self.body.insert(0, p)
        return NO_REWARD, False

    @property
    def heading(self):
        head = self.head
        second = self.second
        if head == second - self.width:
            return 0  # w
        elif head == second - 1:
            return 1  # a
        elif head == second + self.width:
            return 2  # s
        elif head == second + 1:
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
    def second(self):
        return self.body[1]

    @property
    def states(self):
        if self.__states is None:
            body_state = numpy.zeros(self.area, dtype=int)
            for node in self.body:
                body_state[node] = EXIST

            head_state = numpy.zeros(self.area, dtype=int)
            head_state[self.head] = EXIST

            food_state = numpy.zeros(self.area, dtype=int)
            food_state[self.food] = EXIST

            states = numpy.concatenate((body_state, head_state, food_state))
            self.__states = numpy.reshape(states, (CHANNEL, *self.shape))
        return self.__states


EXIST = 1
CHANNEL = 3  # head, body, food

BODY = 0
HEAD = 1
FOOD = 2
