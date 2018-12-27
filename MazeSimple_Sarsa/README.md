# Sarsaによる迷路検索問題
強化学習の学習環境用の迷路探索問題。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向をランダムに選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

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
AGENT_INIT_STATE = 0        # 初期状態の位置 0 ~ 8
BRAIN_LEARNING_RATE = 0.1   # 学習率
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.9          # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果
> 記載中...

![mazesimple_sarsa](https://user-images.githubusercontent.com/25688193/50488132-56b0ec00-0a44-11e9-8efc-341615e7e2ee.gif)<br>

![mazasimple_sarsa_1-1_episode100](https://user-images.githubusercontent.com/25688193/50488170-7c3df580-0a44-11e9-88b1-cfeb3bfaba0e.png)<br>



### ◎ コードの説明

> 記載中...

- 行動価値関数 Q(s,a) を初期化する。<br>
    - 行動価値関数 Q(s,a) は、行を状態 s、列を行動 a とする表形式で実装する。<br>
    - xxx
    ```python
    [SarsaBrain.py]
    ```
2. xxx
