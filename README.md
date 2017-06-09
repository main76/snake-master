# snake-master
reinforcement learning with [Microsoft CNTK](https://github.com/Microsoft/CNTK).

## performance

now, the pretrained model can get an average of 5 scores for each game.

![snake-master](https://master76.github.io/res/snake-master-example.gif)

log:

```plain text
Episode: 1000, Average reward and score for episode: -0.425000, 0.075.
Episode: 2000, Average reward and score for episode: -0.437000, 0.063.
Episode: 3000, Average reward and score for episode: -0.429000, 0.071.
Episode: 4000, Average reward and score for episode: -0.419000, 0.081.
Episode: 5000, Average reward and score for episode: -0.422000, 0.078.
......
Episode: 636000, Average reward and score for episode: 4.838000, 5.338.
Episode: 637000, Average reward and score for episode: 4.797000, 5.297.
Episode: 638000, Average reward and score for episode: 4.620000, 5.120.
Episode: 639000, Average reward and score for episode: 4.812000, 5.312.
Episode: 640000, Average reward and score for episode: 4.712000, 5.212.
```

## text editor

[Microsoft Visual Studio Code](https://github.com/Microsoft/vscode) an awesome editor.

## entry

src/train.py train with 640k episodes.

src/load.py load pretrained model, proceed training.

## dependences

cntk@2.0 cpu-only

python@3.5.2

## devDependences

pygames@1.9.3

yapf@0.16.2
