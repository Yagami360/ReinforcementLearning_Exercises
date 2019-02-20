# ランダムな移動によるFrozenLake
強化学習の学習環境用のスリップなしのFrozenLake。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向をランダムに選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)

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
NUM_EPISODE = 500           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
BRAIN_GAMMDA = 0.99         # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

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
|利得の割引率：`BRAIN_GAMMDA`|0.9|
|利得の設定|ゴール地点(G)：+10<br>穴(H)：-10.0<br>それ以外：0|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`env.seed(8)`|

- 割引利得のエピソード毎の履歴<br>
![frozenlakenotslippery-v0_reward_episode500](https://user-images.githubusercontent.com/25688193/53086043-a47b3980-3547-11e9-87cc-650e5cb7fdd3.png)<br>
