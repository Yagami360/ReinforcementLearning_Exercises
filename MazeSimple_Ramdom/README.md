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
BRAIN_GAMMDA = 0.99         # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果
強化学習の学習環境用の迷路探索問題。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向をランダムに選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|100|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|500|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|利得の設定|ゴール地点：利得＋１、それ以外：利得-0.01|

- 割引利得のエピソード毎の履歴<br>
![mazasimple_ramdom_reward_episode100](https://user-images.githubusercontent.com/25688193/53062593-a1ab2500-3504-11e9-98d3-8ea0e33b43a0.png)<br>

以下のアニメーションは、このランダムな方策に基づく迷路探索問題の探索結果である。エピソードが経過しても、最短ルートでゴールまで到達できるようにはなっていないことが分かる。<br>

- エピソード：0 回目 / 迷路を解くのにかかったステップ数：19<br>
![rl_env_episode0](https://user-images.githubusercontent.com/25688193/53062585-9ce67100-3504-11e9-9317-b1eb28bb7c9f.gif)<br>

- エピソード：25 回目 / 迷路を解くのにかかったステップ数：37<br>
![rl_env_episode25](https://user-images.githubusercontent.com/25688193/53062573-99eb8080-3504-11e9-8652-cf9751e24f94.gif)<br>

- エピソード：50 回目 / 迷路を解くのにかかったステップ数：39<br>
![rl_env_episode50](https://user-images.githubusercontent.com/25688193/53062575-9a841700-3504-11e9-9e11-bdc1dfcfcd20.gif)<br>

<!--
- エピソード：75 回 / 迷路を解くのにかかったステップ数：155<br>
![rl_env_episode75](https://user-images.githubusercontent.com/25688193/53062576-9a841700-3504-11e9-87ea-a5bbc8d2d878.gif)<br>
-->

<!--
- エピソード：100 回 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode99](https://user-images.githubusercontent.com/25688193/53062578-9b1cad80-3504-11e9-988c-b779a2ddb2c1.gif)<br>
-->