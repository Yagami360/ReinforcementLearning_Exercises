# Processing_OpenAIGym
OpenAI Gym での処理フローの練習コード集。（※強化学習のコードではありません。）<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)

## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAI Gym : 0.10.9

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
RL_ENV = "CartPole-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 5          # エピソード試行回数
NUM_TIME_STEP = 100        # １エピソードの時間ステップの最大数
```
