# Deep Q Network（DQN）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] によって、単純な２次元の倒立振子課題を解く。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Deep Q Network（DQN）](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#DeepQNetwork)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9
- PyTorch : 1.0.0

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 500               # エピソード試行回数
NUM_TIME_STEP = 200             # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.0001    # 学習率
BRAIN_BATCH_SIZE = 32           # ミニバッチサイズ
BRAIN_GREEDY_EPSILON = 0.5      # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99             # 利得の割引率
MEMORY_CAPACITY = 10000         # Experience Relay 用の学習用データセットのメモリの最大の長さ
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|
|学習率：`learning_rate`|0.0001|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|
|最適化アルゴリズム|Adam|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|

- 損失関数のグラフ

![cartpole_dqn_1-1_episode500](https://user-images.githubusercontent.com/25688193/52693905-6b712100-2fab-11e9-9862-5f8448c7cf65.png)<br>
> エピソードが経過するにつれて、損失関数の値が０付近の値に向かって収束しており、うまく学習できていることがわかる。<br>

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
※ ポールを左右に振りながらバランスを取るときの振り幅が、Q学習や Sarsa では大きかったのに対して、この DQN では小さい傾向がある？<br>

- エピソード = 10 / 最終時間ステップ数 = 14<br>
![rl_env_cartpole-v0_dqn_episode10](https://user-images.githubusercontent.com/25688193/52695873-eab52380-2fb0-11e9-8b4a-e2fb1c1ac649.gif)<br>

- エピソード = 20 / 最終時間ステップ数 = 15<br>
![rl_env_cartpole-v0_dqn_episode20](https://user-images.githubusercontent.com/25688193/52695903-facd0300-2fb0-11e9-8715-867a02ea8314.gif)<br>

- エピソード = 30 / 最終時間ステップ数 = 57<br>
![rl_env_cartpole-v0_dqn_episode30](https://user-images.githubusercontent.com/25688193/52696008-3ec00800-2fb1-11e9-9881-5bf4c15cd0f0.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 117<br>
![rl_env_cartpole-v0_dqn_episode50](https://user-images.githubusercontent.com/25688193/52695987-34057300-2fb1-11e9-9f83-3772f0beade3.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_dqn_episode100](https://user-images.githubusercontent.com/25688193/52696060-64e5a800-2fb1-11e9-8b29-71d3bd38bf34.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報

