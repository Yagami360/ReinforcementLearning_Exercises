# DQN（2015年NatureバージョンのTargetNetwork使用）によるブロック崩しゲーム（Breakout）
強化学習の学習環境用のブロック崩しゲーム（Breakout）<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2015年NatureバージョンのTarget Q-Network使用）によって、Breakout を解く。<br>

※ ここでの DQN のネットワーク構成は、先の CartPole 問題での簡単のための MLP での実装はなく、本来の CNNで実装している。<br>

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
- OpenAIGym [atari]
- PyTorch : 1.0.0

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 200                   # エピソード試行回数
NUM_TIME_STEP = 1000                # １エピソードの時間ステップの最大数
NUM_NOOP = 30                       # エピソード開始からの何も学習しないステップ数
NUM_SKIP_FRAME = 4                  # スキップするフレーム数
NUM_STACK_FRAME = 1                 # モデルに一度に入力する画像データのフレーム数
BRAIN_LEARNING_RATE = 0.0001        # 学習率
BRAIN_BATCH_SIZE = 32               # ミニバッチサイズ
BRAIN_GREEDY_EPSILON = 0.5          # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99                 # 利得の割引率
MEMORY_CAPACITY = 10000             # Experience Relay 用の学習用データセットのメモリの最大の長さ

```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|200|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|1000|←|
|エピソード開始からの何も学習しないステップ数：`NUM_NOOP`|30|←|
|モデルに一度に入力する画像データのフレーム数：`NUM_STACK_FRAME`|1|4|
|スキップするフレーム数：`NUM_SKIP_FRAME`|4|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam<br>減衰率：`beta1=0.9,beta2=0.999`|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|初期値：0.5（減衰）<br>`ε=0.5*(1/episode+1)`|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|Breakout のデフォルト報酬<br>・下段の青色＆緑色のブロック崩し：１点<br>・中央の黄色＆黄土色のブロック崩し：４点<br>・上段のオレンジ＆赤色のブロック崩し：７点|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|CNN<br>(0): Conv2d(in_channels=**1**, out_channels=32, kernel_size=(8, 8), stride=(4, 4))<br>(1): ReLU()<br>(2): Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))<br>(3): ReLU()<br>(4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))<br>(5): ReLU()<br>(6): Flatten()<br>(7): Linear(in_features=3136, out_features=**4**, bias=True)<br>(8): ReLU()`|←|

<br>

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![BreakoutNoFrameskip-v0_DQN2015_Reward_episode200](https://user-images.githubusercontent.com/25688193/54664421-7f1e2300-4b27-11e9-9c02-82b6349da583.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![BreakoutNoFrameskip-v0_DQN2015_episode200](https://user-images.githubusercontent.com/25688193/54664422-7f1e2300-4b27-11e9-9e14-1f39098044de.png)<br>


以下のアニメーションは、Breakout のブロック崩しを行う様子を示したアニメーションである。<br>
<!--
エピソードの経過と共に、徐々にブロック崩しが出来るようになっており、徐々に学習できていることがわかる。<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 28（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode0](https://user-images.githubusercontent.com/25688193/54664310-31a1b600-4b27-11e9-9ab4-f83ab6b764a3.gif)<br>

- エピソード = 199 / 最終時間ステップ数 = 74（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode199](https://user-images.githubusercontent.com/25688193/54664348-54cc6580-4b27-11e9-9ad4-6cd8acc157eb.gif)<br>
