# Q学習による単純な迷路検索問題
強化学習の学習環境用の単純な迷路探索問題。<br>
価値反復法のアルゴリズムであって、方策オフ型TD制御アルゴリズムである Q学習によって、単純な迷路探索問題を解く。<br>

<!--
単純な迷路探索問題を、Unity ML-Agents のフレームワーク（`Academy`,`Brain`,`Agent`クラス など）を参考にして実装しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>
-->

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Q学習](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Q%E5%AD%A6%E7%BF%92)


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
![mazesimple_sarsa](https://user-images.githubusercontent.com/25688193/50488132-56b0ec00-0a44-11e9-8efc-341615e7e2ee.gif)<br>

- 各状態 S0 ~ S8 での状態価値関数 V(s) のエピソード経過による変化<br>
    ![image](https://user-images.githubusercontent.com/25688193/50521885-4fa0e100-0b0b-11e9-888d-38017892311b.png)<br>
    > ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7）に対応する各状態の状態価値関数の値が、エピソードの経過とともに高い値となっており、うまく価値関数を学習出来ていることが分かる。<br>
    > ※ 尚、状態 S8 は、ゴール状態で行動方策がないため、これに対応する状態価値関数も定義されない。<br>

- Q 学習 と Sarsa での比較<br>
    ![image](https://user-images.githubusercontent.com/25688193/50521914-78c17180-0b0b-11e9-9c2d-5685958b61d3.png)<br>
    > 赤線が Q 学習での状態価値関数 V(s) の変化。青線が、Sarsa での状態価値関数 V(s) の変化。<br>
    > Q 学習のほうが、Sarsa に比べて、収束が早いことが分かる。<br>

### ◎ コードの説明
[Sarsa による単純な迷路検索問題のコード](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/MazeSimple_Sarsa) から、以下の部分を変更するのみ。<br>

- Q 学習
```python
[MazeQlearningBrain.py]
class MazeQlearningBrain( Brain ):
    ...
    def update_q_function( self, state, action, next_state, next_action, reword ):
        # ゴールした場合
        if( next_state == 8 ):
            self._q_function[ state, action ] += self._learning_rate * ( reword - self._q_function[ state, action ] )
        else:
            # Qlearning : self._gamma * np.nanmax( self._q_function[ next_state, : ] )
            # Sarsa : self._gamma * self._q_function[ next_state, action ]
            self._q_function[ state, action ] += self._learning_rate * ( reword + self._gamma * np.nanmax( self._q_function[ next_state, : ] ) - self._q_function[ state, action ] )

        return self._q_function
```

- cf : Sarsa
```python
MazeSarsaBrain.py
class MazeSarsaBrain( Brain ):
    ...
    def update_q_function( self, state, action, next_state, next_action, reword ):
        # ゴールした場合
        if( next_state == 8 ):
            self._q_function[ state, action ] += self._learning_rate * ( reword - self._q_function[ state, action ] )
        else:
            self._q_function[ state, action ] += self._learning_rate * ( reword + self._gamma * self._q_function[ next_state, next_action ] - self._q_function[ state, action ] )

        return self._q_function
```