from agent import Agent
from snake import Handler, CHANNEL
from utils import *
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(abs_path, '..', 'output')

TOTAL_EPISODES = 640000
BATCH_SIZE_BASELINE = 1000

SHAPE = WIDTH, HEIGHT = 10, 10
INPUT_SHAPE, ACTION_COUNT = (CHANNEL, WIDTH, HEIGHT), 3

agent = Agent(INPUT_SHAPE, ACTION_COUNT)
env = Handler(SHAPE, lambda: save_model())

os.makedirs(out_path, exist_ok=True)
os.chdir(out_path)

episode_number = 0
reward_sum = 0

while episode_number < TOTAL_EPISODES:
    reward_sum += run(agent, env)
    episode_number += 1
    if episode_number % BATCH_SIZE_BASELINE == 0:
        log('Episode: %d, Average score for episode %f.' %
              (episode_number, reward_sum / BATCH_SIZE_BASELINE))
        reward_sum = 0
    if episode_number % 10000 == 0:
        save_model('snake.%d.model' % round(episode_number / 10000))
save_model()
