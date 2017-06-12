from utils import INPUT_SHAPE, ACTION_COUNT, SHAPE, Handler
from agent import Agent
import os

EXPECT_SCORES = 16
EXPECT_MOVES = 200
OUTPUT_CASES = 3

abs_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(abs_path, '..', 'screenshots')
model_path = os.path.join(abs_path, '..', 'pretrained', 'snake.10x10_2.56m.model')

agent = Agent(INPUT_SHAPE, ACTION_COUNT, 640000, model_path)
env = Handler(SHAPE)

def start(out_path, infinite_loop = True):
    if infinite_loop:
        global EXPECT_SCORES
        EXPECT_SCORES = float('inf')
    run(agent, env, out_path)

def run(agent, env, out_path):
    s = env.reset()

    while True:
        env.render()

        a = agent.act(s)
        s_, r, done, info = env.step(a)

        if done:  # terminal state
            s_ = None

        agent.observe((s, a, r, s_))
        s = s_

        if done:
            print(info)
            if env.snake.scores >= EXPECT_SCORES and env.snake.total_moves <= EXPECT_MOVES:
                return replay(agent, env, out_path)
            else:
                s = env.reset()

def replay(agent, env, out_path):
    print('Start to replay the chosen one.')
    batch = agent.memory.sample(env.snake.total_moves)
    batch_len = len(batch)

    os.makedirs(out_path, exist_ok=True)
    os.chdir(out_path)

    env.reset() # keep myself happy

    for i in range(batch_len):
        s = batch[i][0]
        env.screenshot(s, '%d.png' % i)
    
    for i in range(3):
        s = batch[-1][0]
        env.screenshot(s, '%d.png' % (i + batch_len))

for i in range(OUTPUT_CASES):
    start(os.path.join(out_path, str(i)))
