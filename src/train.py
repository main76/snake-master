from agent import Agent
from utils import *
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(abs_path, '..', 'output')

agent = Agent(INPUT_SHAPE, ACTION_COUNT)
start(agent, out_path)
