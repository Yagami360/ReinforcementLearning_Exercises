# Sarsaによる倒立振子課題（CartPole）（実装中）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
方策オン型TD制御アルゴリズムである Sarsa によって、単純な２次元の倒立振子課題を解く。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Sarsa](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Sarsa)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 3             # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
NUM_DIZITIZED = 6           # 各状態の離散値への分割数
BRAIN_LEARNING_RATE = 0.1   # 学習率
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99         # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果


### ◎ コードの説明
