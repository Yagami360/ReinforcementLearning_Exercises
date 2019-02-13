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
    1. [【外部リンク】強化学習 / Sarsa](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Sarsa)


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
BRAIN_GAMMDA = 0.9          # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果
|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|100|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|500|
|学習率：`learning_rate`|0.1|
|利得の割引率：`BRAIN_GAMMDA`|0.9|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|

<br>

以下のアニメーションは、Sarsa による迷路探索問題の探索結果である。エピソードが経過するにつれて、うまく最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- エピソード：1 回
![mazesimple_sarsa_episode1](https://user-images.githubusercontent.com/25688193/52710524-df72ef80-2fd2-11e9-95a4-923b5c63a847.gif)<br>

- エピソード：5 回
![mazesimple_sarsa_episode5](https://user-images.githubusercontent.com/25688193/52710562-fa456400-2fd2-11e9-939b-42a0563fc9a3.gif)<br>

- エピソード：10 回
![mazesimple_sarsa_episode10](https://user-images.githubusercontent.com/25688193/52710615-1ba65000-2fd3-11e9-8b53-a3f55e3f632a.gif)<br>

- エピソード：50 回
![mazesimple_sarsa_episode50](https://user-images.githubusercontent.com/25688193/52710669-4395b380-2fd3-11e9-85df-4b9d414ac942.gif)<br>

- エピソード：100 回経過
![mazesimple_sarsa_episode100](https://user-images.githubusercontent.com/25688193/52710334-70959680-2fd2-11e9-92e6-d9d50984eb53.gif)<br>


- 各状態 S0 ~ S8 での状態価値関数 V(s) のエピソード経過による変化<br>

![image](https://user-images.githubusercontent.com/25688193/52710292-565bb880-2fd2-11e9-875d-69882840fd46.png)<br>
> ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7）に対応する各状態の状態価値関数の値が、エピソードの経過とともに高い値となっており、うまく価値関数を学習出来ていることが分かる。<br>
> ※ 尚、状態 S8 は、ゴール状態で行動方策がないため、これに対応する状態価値関数も定義されない。<br>


### ◎ コードの説明

> 記載中...

- 行動価値関数 Q(s,a) を初期化する。<br>
    - 行動価値関数 Q(s,a) は、行を状態 s、列を行動 a とする表形式で実装する。<br>
    - xxx
    ```python
    [SarsaBrain.py]
    ```
2. xxx
