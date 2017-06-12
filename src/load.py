from agent import Agent, SET_MIN_EPSILON
from utils import *
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(abs_path, '..', 'output')
model_path = os.path.join(abs_path, '..', 'pretrained', 'snake.10x10_2.56m.model')

SET_MIN_EPSILON(0)
agent = Agent(INPUT_SHAPE, ACTION_COUNT, 640000, model_path, 0.2)
start(agent, out_path, show=False)
