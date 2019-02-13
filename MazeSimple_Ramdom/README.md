# ランダムな移動による単純な迷路検索問題
強化学習の学習環境用の単純な迷路探索問題。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向をランダムに選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

<!--
単純な迷路探索問題を、Unity ML-Agents のフレームワーク（`Academy`,`Brain`,`Agent`クラス など）を参考にして実装しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>
-->

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
NUM_EPISODE = 1             # エピソード試行回数
NUM_TIME_STEP = 500         # １エピソードの時間ステップの最大数
AGANT_NUM_STATES = 8        # 状態の要素数（s0~s7）※ 終端状態 s8 は除いた数
AGANT_NUM_ACTIONS = 4       # 行動の要素数（↑↓→←）
AGENT_INIT_STATE = 0        # 初期状態の位置 0 ~ 8
BRAIN_GAMMDA = 0.9          # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果
強化学習の学習環境用の迷路探索問題。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向をランダムに選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

以下のアニメーションは、このランダムな方策に基づく迷路探索問題の探索結果である。エピソードが経過しても、最短ルートでゴールまで到達できるようにはなっていないことが分かる。<br>

- エピソード：1 回

![mazesimple_random_episode1](https://user-images.githubusercontent.com/25688193/52712039-c704d400-2fd6-11e9-92e2-e5c6fdaf8599.gif)<br>

- エピソード：50 回

![mazesimple_random_episode50](https://user-images.githubusercontent.com/25688193/52712041-c79d6a80-2fd6-11e9-85ac-7ff4fa809992.gif)<br>

<!--
- エピソード：100 回

![mazesimple_random_episode100](https://user-images.githubusercontent.com/25688193/52712065-de43c180-2fd6-11e9-8314-726205d96c44.gif)<br>
-->
