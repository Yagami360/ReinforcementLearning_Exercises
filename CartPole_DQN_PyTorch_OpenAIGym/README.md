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
NUM_EPISODE = 250               # エピソード試行回数
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

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|250|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|

- 損失関数のグラフ（実行条件１）<br>
![cartpole_dqn_1-1_episode250](https://user-images.githubusercontent.com/25688193/52705363-11318980-2fc6-11e9-943a-055d4961a2cc.png)<br>
> エピソードが経過するにつれて、損失関数の値が０付近の値に向かって収束しており、うまく学習できていることがわかる。<br>

- 損失関数のグラフ（実行条件２）<br>
![cartpole_dqn_1-1_episode500](https://user-images.githubusercontent.com/25688193/52709653-b4879c00-2fd0-11e9-9358-fa7208c0bd71.png)<br>

<!--
![cartpole_dqn_1-1_episode500](https://user-images.githubusercontent.com/25688193/52705418-29090d80-2fc6-11e9-9ccf-eb38bc97e5bc.png)<br>
> 収束していたloss値が突然発散して、その後収束しなくなる？<br>
-->

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
※ ポールを左右に振りながらバランスを取るときの振り幅が、Q学習や Sarsa では大きかったのに対して、この DQN では小さい傾向がある？<br>

- エピソード = 10 / 最終時間ステップ数 = 9<br>
![rl_env_cartpole-v0_dqn_episode10](https://user-images.githubusercontent.com/25688193/52705600-a3d22880-2fc6-11e9-9d0a-4ed5a2d9d285.gif)<br>

- エピソード = 30 / 最終時間ステップ数 = 11<br>
![rl_env_cartpole-v0_dqn_episode30](https://user-images.githubusercontent.com/25688193/52705604-a59bec00-2fc6-11e9-8732-046ad3004d3d.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 17<br>
![rl_env_cartpole-v0_dqn_episode50](https://user-images.githubusercontent.com/25688193/52705608-a896dc80-2fc6-11e9-97d0-1d8b7fb1e8eb.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 51
![rl_env_cartpole-v0_dqn_episode100](https://user-images.githubusercontent.com/25688193/52705611-aaf93680-2fc6-11e9-8d45-c413e69e2574.gif)<br>

- エピソード = 130 / 最終時間ステップ数 = 71<br>
![rl_env_cartpole-v0_dqn_episode130](https://user-images.githubusercontent.com/25688193/52705791-1f33da00-2fc7-11e9-8d64-ff7e8d757cfc.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 189
![rl_env_cartpole-v0_dqn_episode150](https://user-images.githubusercontent.com/25688193/52705651-c95f3200-2fc6-11e9-9497-dee26ee311d3.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 146
![rl_env_cartpole-v0_dqn_episode200](https://user-images.githubusercontent.com/25688193/52705656-ccf2b900-2fc6-11e9-8124-16df0d171e5f.gif)<br>

- エピソード = 230 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_dqn_episode230](https://user-images.githubusercontent.com/25688193/52705664-d0864000-2fc6-11e9-91e0-74b792c51ede.gif)
<br>

- エピソード = 249 / 最終時間ステップ数 = 199
![rl_env_cartpole-v0_dqn_episode250](https://user-images.githubusercontent.com/25688193/52705670-d3813080-2fc6-11e9-8c95-d3046db0e2eb.gif)<br>

### ◎ コードの説明


## ■ デバッグ情報

