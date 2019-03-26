# DQN（2015年バージョンのTargetNetwork使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2015年NatureバージョンのTarget Q-Network使用）によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDQNのネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Deep Q Network（DQN）](http://yagami12.hatenablog.com/entry/2019/02/22/210608#DeepQNetwork)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9
- PyTorch : 1.0.1

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
DEVICE = "GPU"                      # 使用デバイス ("CPU" or "GPU")
RL_ENV = "CartPole-v0"              # 利用する強化学習環境の課題名

NUM_EPISODE = 500                   # エピソード試行回数
NUM_TIME_STEP = 200                 # １エピソードの時間ステップの最大数
NUM_SAVE_STEP = 100                 # 強化学習環境の動画の保存間隔（単位：エピソード数）

BRAIN_LEARNING_RATE = 0.0001        # 学習率
BRAIN_BATCH_SIZE = 32               # ミニバッチサイズ (Default:32)
BRAIN_GREEDY_EPSILON_INIT = 0.5     # ε-greedy 法の ε 値の初期値
BRAIN_GREEDY_EPSILON_FINAL = 0.001  # ε-greedy 法の ε 値の最終値
BRAIN_GREEDY_EPSILON_STEPS = 1000   # ε-greedy 法の ε が減少していくフレーム数
BRAIN_GAMMDA = 0.99                 # 利得の割引率
BRAIN_FREC_TARGET_UPDATE = 20       # Target Network との同期頻度（Default:10_000） 
MEMORY_CAPACITY = 10000             # Experience Relay 用の学習用データセットのメモリの最大の長さ
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|←|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|学習率：`learning_rate`|0.0001|←|
|最適化アルゴリズム|Adam<br>減衰率：`beta1=0.9,beta2=0.999`|←|
|損失関数|smooth L1 関数（＝Huber 関数）|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON_INIT`|1.0|←|
|ε-greedy 法の ε 値の最終値：`BRAIN_GREEDY_EPSILON_FINAL`|0.001|←|
|ε-greedy 法の減衰ステップ数：`BRAIN_GREEDY_EPSILON_STEPS`|5000|←|
|Target Network との同期頻度：`BRAIN_FREC_TARGET_UPDATE`|20|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|MLP（3層）<br>(0): Linear(in_features=4, out_features=32, bias=True)<br>(1): ReLU()<br>(2): Linear(in_features=32, out_features=32, bias=True)<br>(5): ReLU()<br>(6): Linear(in_features=32, out_features=2, bias=True)|MLP（4層）<br>(0): Linear(in_features=4, out_features=32, bias=True)<br>(1): ReLU()<br>(2): Linear(in_features=32, out_features=32, bias=True)<br>(3): ReLU()<br>(4): Linear(in_features=32, out_features=32, bias=True)<br>(5): ReLU()<br>(6): Linear(in_features=32, out_features=2, bias=True)|

<!--
転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|
-->

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![CartPole-v0_DQN2015_Reward_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54997028-26aac200-500e-11e9-9208-698e54dafef9.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![CartPole-v0_DQN2015_Loss_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54997027-26aac200-500e-11e9-9b5f-450418e49b8b.png)<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
![CartPole-v0_DQN2015_Reward_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54995218-8eaad980-5009-11e9-8052-4d21ed5dbcd1.png)<br>

- 損失関数のグラフ（実行条件２）<br>
![CartPole-v0_DQN2015_Loss_episode100_lr0 0001](https://user-images.githubusercontent.com/25688193/54997433-07606480-500f-11e9-8022-21859e3c2743.png)<br>
![CartPole-v0_DQN2015_Loss_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54995214-8d79ac80-5009-11e9-9cec-1d4c58d3a01f.png)<br>


<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
<!--
※ ポールを左右に振りながらバランスを取るときの振り幅が、Q学習や Sarsa では大きかったのに対して、この DQN では小さい傾向がある？<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 10（実行条件２）<br>
![RL_ENV_CartPole-v0_Episode0](https://user-images.githubusercontent.com/25688193/54994281-3b378c00-5007-11e9-8c84-0beaf5ef2410.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 146<br>
![RL_ENV_CartPole-v0_Episode50](https://user-images.githubusercontent.com/25688193/54994282-3bd02280-5007-11e9-93a2-9dc1e378b592.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 199<br>
![RL_ENV_CartPole-v0_Episode100](https://user-images.githubusercontent.com/25688193/54994283-3bd02280-5007-11e9-8b79-ae5a504a092e.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199<br>
![RL_ENV_CartPole-v0_Episode200](https://user-images.githubusercontent.com/25688193/54994290-3d99e600-5007-11e9-9956-f129066c8da3.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199<br>
![RL_ENV_CartPole-v0_Episode300](https://user-images.githubusercontent.com/25688193/54994719-4f2fbd80-5008-11e9-854a-acdd6409425a.gif)<br>
