from agent import Agent
from snake import Handler, CHANNEL
import os
import random

TOTAL_EPISODES = 1000000
BATCH_SIZE_BASELINE = 100

SHAPE = WIDTH, HEIGHT = 10, 10
INPUT_SHAPE, ACTION_COUNT = (CHANNEL, WIDTH, HEIGHT), 3

agent = Agent(INPUT_SHAPE, ACTION_COUNT)
env = Handler(SHAPE, lambda: save_model())


def run(agent):
    s = env.reset()
    R = 0

    while True:
        # env.render()

        a = agent.act(s)
        s_, r, done, info = env.step(a)

        if done:  # terminal state
            s_ = None

        agent.observe((s, a, r, s_))

        if r is not 0:
            if r > 0 or random.random() < 0.1:
                agent.replay(env.moves)

        s = s_
        R += r

        if done:
            print(info)
            return R

def save_model(name='snake.model'):
    agent.brain.model.save(name)

def log(text):
    print(text)
    with open('log.txt', mode='a') as fs:
        fs.write(text)
        fs.write('\n')


abs_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(abs_path, '..', 'output')
os.makedirs(out_path, exist_ok=True)
os.chdir(out_path)

episode_number = 0
reward_sum = 0

while episode_number < TOTAL_EPISODES:
    reward_sum += run(agent)
    episode_number += 1
    if episode_number % BATCH_SIZE_BASELINE == 0:
        log('Episode: %d, Average score for episode %f.' %
              (episode_number, reward_sum / BATCH_SIZE_BASELINE))
        reward_sum = 0
    if episode_number % 10000 == 0:
        save_model('snake.%d.model' % round(episode_number / 10000))
save_model()
