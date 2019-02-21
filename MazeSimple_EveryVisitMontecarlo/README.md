# 逐一訪問モンテカルロ法による単純な迷路検索問題
強化学習の学習環境用の単純な迷路探索問題。<br>
逐一訪問モンテカルロ法によって、単純な迷路探索問題を解く。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / モンテカルロ法](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#%E3%83%A2%E3%83%B3%E3%83%86%E3%82%AB%E3%83%AB%E3%83%AD%E6%B3%95)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 100           # エピソード試行回数
NUM_TIME_STEP = 500         # １エピソードの時間ステップの最大数
AGANT_NUM_STATES = 8        # 状態の要素数（s0~s7）※ 終端状態 s8 は除いた数
AGANT_NUM_ACTIONS = 4       # 行動の要素数（↑↓→←）
AGENT_INIT_STATE = 0        # 初期状態の位置 0 ~ 8
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99          # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|100|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|500|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5（減衰：減衰率11%）|
|利得の設定|ゴール地点：利得+1.0、それ以外：利得-0.01|
|シード値|`np.random.seed(1)`<br>`random.seed(1)`|←|

- 割引利得のエピソード毎の履歴（実行条件１）
![mazasimple_everyvisitmc_reward_episode100](https://user-images.githubusercontent.com/25688193/53144895-6f232a00-35e1-11e9-8eca-ae446c5c5b76.png)<br>
> エピソードが経すると、値が収束しており、うまく学習できていることがわかる。


- 各状態 S0 ~ S8 での状態価値関数 V(s) のエピソード経過による変化（実行条件１）<br>
![image](https://user-images.githubusercontent.com/25688193/53144824-28cdcb00-35e1-11e9-8c8d-cfe81430bb5f.png)<br>
> エピソードが経すると、値が収束しており、うまく学習できていることがわかる。

- 各状態 S0 ~ S8 での行動価値関数 Q(s) の学習完了後のヒートマップ図（実行条件１）<br>
![mazasimple_everyvisitmc_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53144924-86faae00-35e1-11e9-8d89-e2671d466ec3.png)<br>
> 正解ルート（SO→S3→S4→S7→S8）に対応した状態行動対の行動価値関数の値が、大きくなっている（＝小さくなっていない）ことがわかる。

<br>

以下のアニメーションは、逐一訪問モンテカルロ法による迷路探索問題の探索結果である。エピソードが経過するにつれて、うまく最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- エピソード：0 回目 / 迷路を解くのにかかったステップ数：47<br>
![rl_env_episode0](https://user-images.githubusercontent.com/25688193/53144042-f2db1780-35dd-11e9-8404-e657d38490de.gif)<br>

- エピソード：1 回目 / 迷路を解くのにかかったステップ数：13<br>
![rl_env_episode1](https://user-images.githubusercontent.com/25688193/53144193-a5ab7580-35de-11e9-9298-b7d6ea68d905.gif)<br>

- エピソード：2 回目 / 迷路を解くのにかかったステップ数：289<br>
![rl_env_episode2](https://user-images.githubusercontent.com/25688193/53144751-dc828b00-35e0-11e9-8205-e0e3b91c882c.gif)<br>

- エピソード：3 回目 / 迷路を解くのにかかったステップ数：83<br>
![rl_env_episode3](https://user-images.githubusercontent.com/25688193/53144779-ffad3a80-35e0-11e9-8ddc-4f3559e97fa6.gif)<br>

- エピソード：4 回目 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode4](https://user-images.githubusercontent.com/25688193/53144781-0045d100-35e1-11e9-897b-cc782d2d8b72.gif)<br>

- エピソード：5 回目 / 迷路を解くのにかかったステップ数：19<br>
![rl_env_episode5](https://user-images.githubusercontent.com/25688193/53144777-ffad3a80-35e0-11e9-89aa-bf53cc347f66.gif)<br>

- エピソード：10 回目 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode10](https://user-images.githubusercontent.com/25688193/53144431-95e06100-35df-11e9-8cec-4ec82af45f27.gif)<br>

- エピソード：99 回目 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode99](https://user-images.githubusercontent.com/25688193/53144873-5ca8f080-35e1-11e9-9999-ff02b91a3c3a.gif)<br>