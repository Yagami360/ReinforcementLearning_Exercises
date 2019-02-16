# DQN（2013年バージョンのTargetNetwork非使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2013年バージョンのTarget Q-Network非使用）によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDQNのネットワーク構成は、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。

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

|パラメータ名|値（実行条件１）|値（実行条件２）|値（実行条件３）|
|---|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|200|500||
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0に設定|←|転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_dqn2013_reward_episode200](https://user-images.githubusercontent.com/25688193/52898133-b1352000-321d-11e9-9ff2-5ba7b2752648.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![cartpole-v0_dqn2013_1-1_episode200](https://user-images.githubusercontent.com/25688193/52898162-e9d4f980-321d-11e9-9b24-1db551f6b24a.png)<br>
> エピソードが経過するにつれて、損失関数の値が０付近の値に向かって収束しており、うまく学習できていることがわかる。<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
![cartpole-v0_dqn2013_reward_episode500](https://user-images.githubusercontent.com/25688193/52897889-20f5db80-321b-11e9-82c1-50d9796adb9e.png)<br>
- 損失関数のグラフ（実行条件２）<br>
![cartpole-v0_dqn2013_1-1_episode500](https://user-images.githubusercontent.com/25688193/52897890-22bf9f00-321b-11e9-9eb2-d3071e61b570.png)
> 収束していたloss値が突然発散して、その後収束しなくなっており、学習が安定していない。<br>

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
<!--
※ ポールを左右に振りながらバランスを取るときの振り幅が、Q学習や Sarsa では大きかったのに対して、この DQN では小さい傾向がある？<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 11（実行条件１）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/52898182-27398700-321e-11e9-96c4-738a7382eb38.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 48<br>
![rl_env_cartpole-v0_episode50](https://user-images.githubusercontent.com/25688193/52898183-27d21d80-321e-11e9-807a-0f3158bde4a3.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 80<br>
![rl_env_cartpole-v0_episode199](https://user-images.githubusercontent.com/25688193/52898186-27d21d80-321e-11e9-937d-954d749859c8.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 116<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/52898184-27d21d80-321e-11e9-95f5-0d1e9c1a6017.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 180<br>
![rl_env_cartpole-v0_episode150](https://user-images.githubusercontent.com/25688193/52898185-27d21d80-321e-11e9-8e49-d3917aec6c98.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報

