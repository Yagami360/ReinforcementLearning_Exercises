# DDQNによる倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDDQN [Double-DQN] によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDDQNのネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Double-DQN（DDQN）](http://yagami12.hatenablog.com/entry/2019/02/22/210608#Double-DQN%EF%BC%88DDQN%EF%BC%89)


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

- 割引利得のエピソード毎の履歴（実行条件１）<br>
<!--
![cartpole-v0_reward_episode500](https://user-images.githubusercontent.com/25688193/53781928-4e969080-3f4e-11e9-8b97-a693e3c4e3cc.png)<br>
-->

- 損失関数のグラフ（実行条件１）<br>
<!--
![cartpole-v0_loss_episode500](https://user-images.githubusercontent.com/25688193/53781929-4e969080-3f4e-11e9-9671-9c7d5ea6ad40.png)<br>
> 通常の DQN より、学習が安定化されていることがわかる。<br>
-->

<br>

以下のアニメーションは、本アルゴリズムにより、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>

<!--
- エピソード = 0 / 最終時間ステップ数 = 10（実行条件１）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/53781816-bef0e200-3f4d-11e9-9cc8-17c767f10f88.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 28（実行条件１）<br>
![rl_env_cartpole-v0_episode50](https://user-images.githubusercontent.com/25688193/53781817-bef0e200-3f4d-11e9-9353-3f126292bd02.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 23（実行条件１）<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/53781815-be584b80-3f4d-11e9-8d02-f6304c2a7bae.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 199（実行条件１）<br>
![rl_env_cartpole-v0_episode150](https://user-images.githubusercontent.com/25688193/53781862-f069ad80-3f4d-11e9-8384-0b3e5b61a678.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199（実行条件１）<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/53781821-c1533c00-3f4d-11e9-8b30-7fbae05ea8ab.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199（実行条件１）<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/53781828-c3b59600-3f4d-11e9-8b2d-972ea8695994.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 199（実行条件１）<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/53781829-c6b08680-3f4d-11e9-82bd-03e1f746b158.gif)<br>

- エピソード = 500 / 最終時間ステップ数 = 199（実行条件１）<br>
<br>
-->