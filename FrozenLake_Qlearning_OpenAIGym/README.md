# Q学習によるFrozenLake
強化学習の学習環境用のスリップなしのFrozenLake。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向をランダムに選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Q学習](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Q%E5%AD%A6%E7%BF%92)

## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
RL_ENV = "FrozenLakeNotSlippery-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 500           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.1   # 学習率
BRAIN_GREEDY_EPSILON = 0.01  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99          # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果
<!--
このFrozenLakeでは、床が凍っているので進みたい方向に確実に進めるわけではありません。
進みたい方向へは1/3の確率でしか進めず、残り1/3ずつの確率で進む方向が90度変わります。例えば、下を選択した場合、1/3で下に進み、1/3で右、1/3で左に進みます。<br>
-->

|||||FrozenLake|
|---|---|---|---|---|
|S|F|F|F|(S: starting point, safe)|
|F|H|F|H|(F: frozen surface, safe)|
|F|F|F|H|(H: hole, fall to your doom)|
|H|F|F|G|(G: goal, where the frisbee is located)|

<br>

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.01（固定）|
|利得の設定|ゴール地点(G)：+10<br>穴(H)：-10.0<br>それ以外：0|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`env.seed(8)`|

- 割引利得のエピソード毎の履歴<br>
![frozenlakenotslippery-v0_reward_episode500](https://user-images.githubusercontent.com/25688193/53085869-359de080-3547-11e9-946b-7ee691362a53.png)<br>
> エピソード経過時の収益の落ち込みは、ε-greedy 法の ε 値での落ち込み

- Q関数のヒートマップ<br>
![frozenlakenotslippery-v0_qlearning_qfunction_episode500](https://user-images.githubusercontent.com/25688193/53085870-359de080-3547-11e9-9f8a-3fba9cf7168f.png)<br>
