# snake-master
reinforcement learning with [Microsoft CNTK](https://github.com/Microsoft/CNTK).

## performance

now, the pretrained model can get an average of 7.7 scores for each game.

![snake-master](https://master76.github.io/res/snake-master-example.gif)

log:

```plain text
Episode: 10000, Average reward and score for episode: -0.424200, 0.076.
Episode: 20000, Average reward and score for episode: -0.417600, 0.082.
Episode: 30000, Average reward and score for episode: -0.415300, 0.085.
Episode: 40000, Average reward and score for episode: -0.415300, 0.085.
Episode: 50000, Average reward and score for episode: -0.416800, 0.083.
......
Episode: 2520000, Average reward and score for episode: 7.137500, 7.638.
Episode: 2530000, Average reward and score for episode: 7.141700, 7.642.
Episode: 2540000, Average reward and score for episode: 7.214600, 7.715.
Episode: 2550000, Average reward and score for episode: 7.213900, 7.714.
Episode: 2560000, Average reward and score for episode: 7.105500, 7.606.
```

## text editor

[Visual Studio Code](https://github.com/Microsoft/vscode) an awesome editor.

## entry

src/train.py train with 640k episodes.

src/load.py load pretrained model, proceed training.

src/pref.py load pretrained model, show how it acts. (requires pygame)

## dependences

cntk@2.0 cpu-only

python@3.5.2

## devDependences

pygames@1.9.3

yapf@0.16.2
