# DQN（2015年NatureバージョンのTargetNetwork使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2015年NatureバージョンのTarget Q-Network使用）によって、単純な２次元の倒立振子課題を解く。<br>

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

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|xxx|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|

- 損失関数のグラフ（実行条件１）<br>
![cartpole-v0_dqn2015_1-1_episode500](https://user-images.githubusercontent.com/25688193/52782576-c3cf1e00-3092-11e9-8183-6e4de8363279.png)<br>


<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>

- エピソード = 10 / 最終時間ステップ数 = 9<br>
![rl_env_cartpole-v0_dqn2015_episode10](https://user-images.githubusercontent.com/25688193/52781450-f3305b80-308f-11e9-8726-1115c839d069.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 10<br>
![rl_env_cartpole-v0_dqn2015_episode50](https://user-images.githubusercontent.com/25688193/52781455-f62b4c00-308f-11e9-8adc-84d46f857286.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 116<br>
![rl_env_cartpole-v0_dqn2015_episode100](https://user-images.githubusercontent.com/25688193/52781458-f9263c80-308f-11e9-9709-60a64387ce8a.gif)<br>

- エピソード = 130 / 最終時間ステップ数 = 80<br>
![rl_env_cartpole-v0_dqn2015_episode130](https://user-images.githubusercontent.com/25688193/52781606-54582f00-3090-11e9-84ee-1d7bc03d7d25.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_dqn2015_episode150](https://user-images.githubusercontent.com/25688193/52781462-fb889680-308f-11e9-80e8-24a8ecf031f2.gif)<br>

- エピソード = 180 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_dqn2015_episode180](https://user-images.githubusercontent.com/25688193/52781702-8d909f00-3090-11e9-8d20-3d111d8c4c30.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_dqn2015_episode200](https://user-images.githubusercontent.com/25688193/52781465-fdeaf080-308f-11e9-929a-fc507b6595f7.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報

