import random
import os
from snake import Handler, CHANNEL

SHAPE = WIDTH, HEIGHT = 10, 10
INPUT_SHAPE, ACTION_COUNT = (CHANNEL, WIDTH, HEIGHT), 3
CHECKPOINT_FREQ = 80000

def start(agent, out_path, TOTAL_EPISODES=640000, BATCH_SIZE_BASELINE=10000, show=False):
    os.makedirs(out_path, exist_ok=True)
    os.chdir(out_path)

    env = Handler(SHAPE, lambda: save_model(agent))

    episode_number = agent.steps
    reward_sum = 0
    score_sum = 0

    while episode_number < TOTAL_EPISODES:
        reward, score = run(agent, env, show)
        reward_sum += reward
        score_sum += score
        episode_number += 1
        if episode_number % BATCH_SIZE_BASELINE == 0:
            log('Episode: %d, Average reward and score for episode: %f, %.3f.' %
                (episode_number, reward_sum / BATCH_SIZE_BASELINE, score_sum / BATCH_SIZE_BASELINE))
            reward_sum = 0
            score_sum = 0
        if episode_number % CHECKPOINT_FREQ == 0:
            save_model(agent, 'snake.%d.model' % round(episode_number / CHECKPOINT_FREQ))
    save_model(agent)

def run(agent, env, show=False):
    s = env.reset()
    R = 0

    while True:
        if show:
            env.render()

        a = agent.act(s)
        s_, r, done, info = env.step(a)

        if done:  # terminal state
            s_ = None

        agent.observe((s, a, r, s_))

        s = s_
        R += r

        if done:
            if r > 0 or random.random() > agent.epsilon - 0.1:
                agent.replay(env.moves)
            print(info)
            return R, env.snake.scores

def save_model(agent, name='snake.model'):
    agent.brain.model.save(name)

def log(text):
    print(text)
    with open('log.txt', mode='a') as fs:
        fs.write(text)
        fs.write('\n')
