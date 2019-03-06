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
|エピソード試行回数：`NUM_EPISODE`|500|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5（減衰）|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>出力層：行動数（2）|MLP（4層）<br>入力層：状態数（4）<br>隠れ層１：32ノード<br>隠れ層２：32ノード<br>出力層：行動数（2）|


- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_reward_episode500](https://user-images.githubusercontent.com/25688193/53781928-4e969080-3f4e-11e9-8b97-a693e3c4e3cc.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![cartpole-v0_loss_episode500](https://user-images.githubusercontent.com/25688193/53781929-4e969080-3f4e-11e9-9671-9c7d5ea6ad40.png)<br>

> 通常の DQN より、学習が安定化されていることがわかる。<br>

<br>

以下のアニメーションは、本アルゴリズムにより、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>

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

### ◎ コードの説明


## ■ デバッグ情報

```python
a_m : tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0])
a_m : tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1])

a_m_non_final_next_states : tensor([[1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1]])

next_outputs : tensor([[-0.1413,  0.0316],
        [-0.1261,  0.0474],
        [-0.1609,  0.0018],
        [-0.2363,  0.0498],
        [-0.2587,  0.0724],
        [-0.2096,  0.0230],
        [-0.1899,  0.0059],
        [-0.1933,  0.0097],
        [-0.3051,  0.1412],
        [-0.2574,  0.0723],
        [-0.2344,  0.0445],
        [-0.2023,  0.0202],
        [-0.1230,  0.0511],
        [-0.2395,  0.0533],
        [-0.2335,  0.0447],
        [-0.2625,  0.0776],
        [-0.2085,  0.0243],
        [-0.1424,  0.0299],
        [-0.1247,  0.0490],
        [-0.1664,  0.0004],
        [-0.1849,  0.0034],
        [-0.2039,  0.0211],
        [-0.1437,  0.0285],
        [-0.2770,  0.0953],
        [-0.2295,  0.0451],
        [-0.2450,  0.0592],
        [-0.2910,  0.1186],
        [-0.2103,  0.0224],
        [-0.2336,  0.0483],
        [-0.1686,  0.0139]], grad_fn=<AddmmBackward>)

next_state_values : 
    tensor([0.0316, 0.0474, 0.0018, 0.0498, 0.0724, 0.0230, 0.0059, 0.0097, 0.1412, 0.0723, 0.0445, 0.0202, 0.0511, 0.0533, 0.0447, 0.0776, 0.0243, 0.0299, 0.0490, 0.0004, 0.0034, 0.0000, 0.0211, 0.0285, 0.0953, 0.0451, 0.0592, 0.0000, 0.1186, 0.0224, 0.0483, 0.0139])

```