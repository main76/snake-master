import random

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

        if r is not 0:
            if r > 0 or random.random() > agent.epsilon - 0.1:
                agent.replay(env.moves)

        s = s_
        R += r

        if done:
            print(info)
            return R

def save_model(agent, name='snake.model'):
    agent.brain.model.save(name)

def log(text):
    print(text)
    with open('log.txt', mode='a') as fs:
        fs.write(text)
        fs.write('\n')
