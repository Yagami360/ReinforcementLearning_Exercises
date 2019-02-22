# Sarsaによる単純な迷路検索問題
強化学習の学習環境用の単純な迷路探索問題。<br>
方策オン型TD制御アルゴリズムである Sarsa によって、単純な迷路探索問題を解く。<br>

<!--
単純な迷路探索問題を、Unity ML-Agents のフレームワーク（`Academy`,`Brain`,`Agent`クラス など）を参考にして実装しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>
-->

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Sarsa](http://yagami12.hatenablog.com/entry/2019/02/22/210608#Sarsa)


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
BRAIN_LEARNING_RATE = 0.1   # 学習率
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
|学習率：`learning_rate`|0.1|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5（減衰）|
|利得の設定|ゴール地点：利得+1.0、それ以外：利得-0.01|
|シード値|`np.random.seed(1)`<br>`random.seed(1)`|←|

<br>

- 割引利得のエピソード毎の履歴<br>
![mazasimple_sarsa_reward_episode100](https://user-images.githubusercontent.com/25688193/53059008-b1bd0780-34f8-11e9-926f-3be7a6262b8e.png)<br>

- 各状態 S0 ~ S8 での状態価値関数 V(s) のエピソード経過による変化<br>
![image](https://user-images.githubusercontent.com/25688193/53060285-445fa580-34fd-11e9-86d2-6c63e544ffea.png)<br>

> ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7）に対応する各状態の状態価値関数の値が、エピソードの経過とともに高い値となっており、うまく価値関数を学習出来ていることが分かる。<br>
> ※ 尚、状態 S8 は、ゴール状態で行動方策がないため、これに対応する状態価値関数も定義されない。<br>

- 各状態 {S0 ~ S8}、各行動 {↑、→、↓、←} での行動価値関数 Q(s) の学習完了後のヒートマップ図（実行条件１）<br>
![mazasimple_sarsa_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53061348-d0bf9780-3500-11e9-9f40-26e75a6ace15.png)<br>
> 上図は、下表のように、各状態のセルを９つのグリッドに分割し、それぞれ↑→↓←での状態行動対に対する行動価値関数をヒートマップで表示した図である。

|S0|↑||S1|↑||S2|↑||
|---|---|---|---|---|---|---|---|---|
|←|平均値|→|←|平均値|→|←|平均値|→|
||↓|||↓|||↓||
|S3|↑||S4|↑||S5|↑||
|←|平均値|→|←|平均値|→|←|平均値|→|
||↓|||↓|||↓||
|S6|↑||S7|↑||S8|↑||
|←|平均値|→|←|平均値|→|←|平均値|→|
||↓|||↓|||↓||

> ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7 → S8）に対応する状態行動対の行動価値関数の値が、高い値となっており、うまく行動価値関数を学習出来ていることが分かる。<br>
> ※ 尚、終端状態 S8 の行動価値関数は常に０の値となる。<br>


<br>

以下のアニメーションは、Sarsa による迷路探索問題の探索結果である。エピソードが経過するにつれて、うまく最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- エピソード：1 回 / 迷路を解くのにかかったステップ数：47<br>
![rl_env_episode0](https://user-images.githubusercontent.com/25688193/53061281-aa016100-3500-11e9-85ed-9226028d930b.gif)<br>

- エピソード：5 回 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode5](https://user-images.githubusercontent.com/25688193/53061279-a8d03400-3500-11e9-8cbd-31b8ad88cc9e.gif)<br>

- エピソード：10 回 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode10](https://user-images.githubusercontent.com/25688193/53061280-a968ca80-3500-11e9-82bd-157997224fef.gif)<br>

- エピソード：100 回経過 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode99](https://user-images.githubusercontent.com/25688193/53061293-ac63bb00-3500-11e9-8994-104274e6cf0c.gif)<br>


### ◎ コードの説明

> 記載中...

- 行動価値関数 Q(s,a) を初期化する。<br>
    - 行動価値関数 Q(s,a) は、行を状態 s、列を行動 a とする表形式で実装する。<br>
    - xxx
    ```python
    [SarsaBrain.py]
    ```
2. xxx
