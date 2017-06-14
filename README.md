# snake-master
reinforcement learning with [Microsoft CNTK](https://github.com/Microsoft/CNTK).

## latest commits

add the heading information to the input features, now the training is more efficient.

## performance

now, the pretrained model can get an average of 7.7 scores for each game.

![snake-master](https://master76.github.io/res/snake-master-example.gif)

old log:

```plaintext
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

new one:

```plaintext
Episode: 600000, Average reward and score for episode: 6.429200, 6.929.
Episode: 610000, Average reward and score for episode: 6.677300, 7.177.
Episode: 620000, Average reward and score for episode: 6.735800, 7.236.
Episode: 630000, Average reward and score for episode: 6.844000, 7.344.
Episode: 640000, Average reward and score for episode: 6.891000, 7.391.
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
