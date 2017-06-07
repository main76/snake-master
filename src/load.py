from agent import Agent
from utils import *
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(abs_path, '..', 'output')
model_path = os.path.join(abs_path, '..', 'pretrained', 'snake.10x10_640k.model')

agent = Agent(INPUT_SHAPE, ACTION_COUNT, 640000, model_path)
start(agent, out_path, show=True)
