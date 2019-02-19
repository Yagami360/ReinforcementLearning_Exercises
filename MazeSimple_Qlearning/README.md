# Q学習による単純な迷路検索問題
強化学習の学習環境用の単純な迷路探索問題。<br>
方策オフ型TD制御アルゴリズムである Q学習によって、単純な迷路探索問題を解く。<br>

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
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|

- 割引利得のエピソード毎の履歴（実行条件１）
![mazasimple_qlearning_reward_episode100](https://user-images.githubusercontent.com/25688193/53012236-c1e7cf00-3485-11e9-9813-71bf8bd5cc2d.png)<br>

- 各状態 S0 ~ S8 での状態価値関数 V(s) のエピソード経過による変化（実行条件１）<br>
<!--
![image](https://user-images.githubusercontent.com/25688193/52710903-e2baab00-2fd3-11e9-82f6-3d00011914ec.png)<br>
-->
![image](https://user-images.githubusercontent.com/25688193/53012133-7e8d6080-3485-11e9-89d2-8147ccbe05ee.png)<br>
> ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7）に対応する各状態の状態価値関数の値が、エピソードの経過とともに高い値となっており、うまく価値関数を学習出来ていることが分かる。<br>
> ※ 尚、終端状態 S8 の状態価値関数は常に０の値となる。<br>

- 各状態 S0 ~ S8 での行動価値関数 Q(s) の学習完了後のヒートマップ図（実行条件１）<br>
![mazasimple_qlearning_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53012335-07a49780-3486-11e9-8659-9ad6c338ed36.png)<br>

> ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7）に対応する状態行動対の行動価値関数の値が、高い値となっており、うまく行動価値関数を学習出来ていることが分かる。<br>
> ※ 尚、終端状態 S8 の行動価値関数は常に０の値となる。<br>

- Q 学習 と Sarsa での比較<br>
<!--
![image](https://user-images.githubusercontent.com/25688193/52710776-848dc800-2fd3-11e9-87ba-d30f3a96aeeb.png)<br>
-->
![mazasimple_qlearning-sarsa_reward_episode100](https://user-images.githubusercontent.com/25688193/53013172-6e2ab500-3488-11e9-9251-59ae659edcec.png)<br>
![image](https://user-images.githubusercontent.com/25688193/53013122-4d625f80-3488-11e9-8c46-b9d7b39996a7.png)<br>
![mazasimple_qlearnig_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53013276-b6e26e00-3488-11e9-9cd1-0d1af4e1a839.png)<br>
![mazasimple_sarsa_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53013262-a7632500-3488-11e9-8e71-f9a86a50df32.png)<br>

> 赤線が Q 学習での変化。青線が、Sarsa での変化。<br>
> Q 学習のほうが、Sarsa に比べて、落ち込みが少なく、収束が早い傾向が見てとれる。<br>

<br>

以下のアニメーションは、Q学習による迷路探索問題の探索結果である。エピソードが経過するにつれて、うまく最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- エピソード：0 回目 / 迷路を解くのにかかったステップ数：47<br>
![rl_env_episode0](https://user-images.githubusercontent.com/25688193/53012366-21de7580-3486-11e9-9d57-9a64de9611e1.gif)<br>

- エピソード：5 回目 / 迷路を解くのにかかったステップ数：21<br>
![rl_env_episode5](https://user-images.githubusercontent.com/25688193/53012367-21de7580-3486-11e9-84db-3508510d5527.gif)<br>

- エピソード：10 回目 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode10](https://user-images.githubusercontent.com/25688193/53012364-2145df00-3486-11e9-93a7-7b2aba5c1ce4.gif)
<br>

- エピソード：100 回目経過 / 迷路を解くのにかかったステップ数：5
![rl_env_episode99](https://user-images.githubusercontent.com/25688193/53012529-8bf71a80-3486-11e9-9671-3eb905317687.gif)<br>


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
[MazeSarsaBrain.py]
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