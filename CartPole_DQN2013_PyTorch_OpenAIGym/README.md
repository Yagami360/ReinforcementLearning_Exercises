# DQN（2013年バージョンのTargetNetwork非使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2013年バージョンのTarget Q-Network非使用）によって、単純な２次元の倒立振子課題を解く。<br>

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
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|MLP（3層）<br>(0): Linear(in_features=4, out_features=32, bias=True)<br>(1): ReLU()<br>(2): Linear(in_features=32, out_features=32, bias=True)<br>(5): ReLU()<br>(6): Linear(in_features=32, out_features=2, bias=True)|MLP（4層）<br>(0): Linear(in_features=4, out_features=32, bias=True)<br>(1): ReLU()<br>(2): Linear(in_features=32, out_features=32, bias=True)<br>(3): ReLU()<br>(4): Linear(in_features=32, out_features=32, bias=True)<br>(5): ReLU()<br>(6): Linear(in_features=32, out_features=2, bias=True)|

<!--
転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|
-->

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![CartPole-v0_DQN2013_Reward_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/55083350-bf147580-50e6-11e9-9ca0-08f2c85a3916.png)<br>
<!--
![CartPole-v0_DQN2013_Reward_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54998197-e0a32d80-5010-11e9-890a-61542dfc6fb6.png)<br>
-->

- 損失関数のグラフ（実行条件１）<br>
![CartPole-v0_DQN2013_Loss_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/55083349-bf147580-50e6-11e9-8b98-cee1e1ec42e6.png)<br>
<!--
![CartPole-v0_DQN2013_Loss_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54998194-e0a32d80-5010-11e9-9dcd-15e0ab566bb6.png)<br>
-->

<!--
> 途中で損失関数の値が発散しており、その後０付近の収束しておらず、うまく学習できていないことがわかる。<br>
-->

<!--
- 割引利得のエピソード毎の履歴（実行条件２）<br>
![CartPole-v0_DQN2013_Reward_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54996397-84d6a580-500c-11e9-9b2b-4e0116d9eab1.png)<br>
-->

<!--
- 損失関数のグラフ（実行条件２）<br>
![CartPole-v0_DQN2013_Loss_episode100_lr0 0001](https://user-images.githubusercontent.com/25688193/54996753-7b9a0880-500d-11e9-8efc-9c0bb6fb3231.png)<br>
![CartPole-v0_DQN2013_Loss_episode500_lr0 0001](https://user-images.githubusercontent.com/25688193/54996396-84d6a580-500c-11e9-82b8-86af38a9ac1e.png)<br>
-->

<!--
> エピソードが経過するにつれて、損失関数の値が０付近の値に向かって収束しており、うまく学習できていることがわかる。<br>
> また、実行条件１より、学習が安定化していることがわかる。（実行条件１のMLPより、層数が多いため？）<br>
-->

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
<!--
※ ポールを左右に振りながらバランスを取るときの振り幅が、Q学習や Sarsa では大きかったのに対して、この DQN では小さい傾向がある？<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 14（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode0](https://user-images.githubusercontent.com/25688193/55083446-e703d900-50e6-11e9-8546-99067777b120.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 63（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode50](https://user-images.githubusercontent.com/25688193/55083449-e79c6f80-50e6-11e9-9674-8ee1f5fa1b33.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 199（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode100](https://user-images.githubusercontent.com/25688193/55083452-e79c6f80-50e6-11e9-9aa6-942b2ed35734.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode200](https://user-images.githubusercontent.com/25688193/55083455-e8350600-50e6-11e9-95bd-9d98bdb83f8d.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode300](https://user-images.githubusercontent.com/25688193/55083457-e8350600-50e6-11e9-8506-6ecab96030a1.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 199（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode400](https://user-images.githubusercontent.com/25688193/55083460-e8cd9c80-50e6-11e9-9ce4-c3e77b9a27a3.gif)<br>

- エピソード = 500 / 最終時間ステップ数 = xxx（実行条件１）<br>
![RL_ENV_CartPole-v0_Episode499](https://user-images.githubusercontent.com/25688193/55083462-e8cd9c80-50e6-11e9-9ef3-8333b86e9e35.gif)<br>
