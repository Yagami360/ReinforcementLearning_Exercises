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
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|
|利得の設定|ゴール地点：利得+1.0、それ以外：利得-0.01|
|シード値|`np.random.seed(1)`<br>`random.seed(1)`|←|

- 割引利得のエピソード毎の履歴（実行条件１）
![mazasimple_qlearning_reward_episode100](https://user-images.githubusercontent.com/25688193/53015127-9e288700-348d-11e9-9abb-0adc663cb1d4.png)<br>

- 各状態 S0 ~ S8 での状態価値関数 V(s) のエピソード経過による変化（実行条件１）<br>
![image](https://user-images.githubusercontent.com/25688193/53015105-8e10a780-348d-11e9-8a08-81be2b5a05f0.png)<br>
> ゴールへたどり着くための正解ルート（S0 → S3 → S4 → S7）に対応する各状態の状態価値関数の値が、エピソードの経過とともに高い値となっており、うまく価値関数を学習出来ていることが分かる。<br>
> ※ 尚、終端状態 S8 の状態価値関数は常に０の値となる。<br>

- 各状態 S0 ~ S8 での行動価値関数 Q(s) の学習完了後のヒートマップ図（実行条件１）<br>
![mazasimple_qlearning_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53062093-02d1f900-3503-11e9-989f-875de6ad1fbb.png)<br>
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

- Q 学習 と Sarsa での比較<br>
![mazasimple_qlearning-sarsa_reward_episode100](https://user-images.githubusercontent.com/25688193/53014847-fe6af900-348c-11e9-84cf-c46694d61daa.png)<br>
![image](https://user-images.githubusercontent.com/25688193/53014893-278b8980-348d-11e9-9ec9-ae9cb994deee.png)<br>
![mazasimple_qlearning_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53062204-59d7ce00-3503-11e9-9f98-6169336b9552.png)<br>
![mazasimple_sarsa_qfunction_episode100](https://user-images.githubusercontent.com/25688193/53062205-5b08fb00-3503-11e9-91fc-06906c57e5aa.png)<br>

> 赤線が Q 学習での変化。青線が、Sarsa での変化。<br>
> Q 学習のほうが、Sarsa に比べて、落ち込みが少なく、収束が早い傾向が見てとれる。<br>

<br>

以下のアニメーションは、Q学習による迷路探索問題の探索結果である。エピソードが経過するにつれて、うまく最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- エピソード：0 回目 / 迷路を解くのにかかったステップ数：45<br>
![rl_env_episode0](https://user-images.githubusercontent.com/25688193/53061973-98b95400-3502-11e9-9ae2-9c005ed9bf88.gif)<br>

- エピソード：5 回目 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode5](https://user-images.githubusercontent.com/25688193/53061974-98b95400-3502-11e9-8f21-ed80532218c0.gif)<br>

- エピソード：10 回目 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode10](https://user-images.githubusercontent.com/25688193/53061972-9820bd80-3502-11e9-8822-d62570e7d9a6.gif)<br>

- エピソード：100 回目経過 / 迷路を解くのにかかったステップ数：5
![rl_env_episode100](https://user-images.githubusercontent.com/25688193/53062044-d6b67800-3502-11e9-9c57-5e98214069a5.gif)<br>


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