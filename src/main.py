from agent import Agent
from snake import Handler
import os

TOTAL_EPISODES = 640000
BATCH_SIZE_BASELINE = 50

WIDTH, HEIGHT = 10, 10
STATE_COUNT, ACTION_COUNT = WIDTH * HEIGHT, 3

agent = Agent(STATE_COUNT, ACTION_COUNT)
env = Handler(WIDTH, HEIGHT, lambda: save_model())


def run(agent):
    s = env.reset()
    R = 0

    while True:
        env.render()

        a = agent.act(s)
        s_, r, done, info = env.step(a)

        if done:  # terminal state
            s_ = None

        agent.observe((s, a, r, s_))
        agent.replay(env.moves)

        s = s_
        R += r

        if done:
            print(info)
            return R


def save_model(name='snake.model'):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(abs_path, '..', 'output')
    os.makedirs(out_path, exist_ok=True)
    os.chdir(out_path)
    agent.brain.model.save(name)


episode_number = 0
reward_sum = 0

while episode_number < TOTAL_EPISODES:
    reward_sum += run(agent)
    episode_number += 1
    if episode_number % BATCH_SIZE_BASELINE == 0:
        print('Episode: %d, Average score for episode %f.' %
              (episode_number, reward_sum / BATCH_SIZE_BASELINE))
        reward_sum = 0
save_model()