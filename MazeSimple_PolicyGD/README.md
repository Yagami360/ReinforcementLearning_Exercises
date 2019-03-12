# 方策勾配法による単純な迷路検索問題
強化学習の学習環境用の単純な迷路探索問題。<br>
方策勾配法によって、単純な迷路探索問題を解く。<br>

<!--
単純な迷路探索問題を、Unity ML-Agents のフレームワーク（`Academy`,`Brain`,`Agent`クラス など）を参考にして実装しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>
-->

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / 方策勾配に基づく強化学習手法](http://yagami12.hatenablog.com/entry/2019/02/22/210608#%E6%96%B9%E7%AD%96%E5%8B%BE%E9%85%8D%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E6%89%8B%E6%B3%95)

## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1

## ■ 使用法

- 実行方法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 5000          # エピソード試行回数
NUM_TIME_STEP = 500         # １エピソードの時間ステップの最大数
AGANT_NUM_STATES = 8        # 状態の要素数（s0~s7）※ 終端状態 s8 は除いた数
AGANT_NUM_ACTIONS = 4       # 行動の要素数（↑↓→←）
AGENT_INIT_STATE = 0        # 初期状態の位置 0 ~ 8
BRAIN_LEARNING_RATE = 0.1   # 学習率
BRAIN_GAMMDA = 0.99         # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果
方策反復法の具体例なアルゴリズムの１つである方策勾配法によって、迷路探索問題を解く。<br>

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|5000|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|500|
|学習率：`learning_rate`|0.1|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|利得の設定|ゴール地点：利得＋１、それ以外：利得-0.01|
|シード値|`np.random.seed(1)`<br>`random.seed(1)`|←|

- 割引利得のエピソード毎の履歴<br>
![mazasimple_policygd_reward_episode5000_](https://user-images.githubusercontent.com/25688193/53063294-f354af00-3506-11e9-9ce2-233a7f3d7c27.png)<br>

<br>

以下のアニメーションは、方策勾配法による迷路探索問題の探索結果である。エピソードが経過するにつれて、うまく最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- エピソード：1 回 / 迷路を解くのにかかったステップ数：87<br>
![rl_env_episode0](https://user-images.githubusercontent.com/25688193/53062983-f00cf380-3505-11e9-9352-6592f4ca39ec.gif)<br>

- エピソード：100 回 / 迷路を解くのにかかったステップ数：7<br>
![rl_env_episode100](https://user-images.githubusercontent.com/25688193/53063084-4712c880-3506-11e9-98e6-35e106b2c1c8.gif)<br>

- エピソード：200 回 / 迷路を解くのにかかったステップ数：23<br>
![rl_env_episode200](https://user-images.githubusercontent.com/25688193/53063085-47ab5f00-3506-11e9-8bd4-505738df0478.gif)<br>

- エピソード：300 回 / 迷路を解くのにかかったステップ数：11<br>
![rl_env_episode300](https://user-images.githubusercontent.com/25688193/53063090-4bd77c80-3506-11e9-9db5-4c2f04ab5cbc.gif)<br>

- エピソード：400 回 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode400](https://user-images.githubusercontent.com/25688193/53063096-50039a00-3506-11e9-9b09-c2c1b5b6c348.gif)<br>

- エピソード：500 回 / 迷路を解くのにかかったステップ数：7<br>
![rl_env_episode500](https://user-images.githubusercontent.com/25688193/53063098-5134c700-3506-11e9-8088-a1893645c709.gif)<br>

- エピソード：1000 回 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode1000](https://user-images.githubusercontent.com/25688193/53063145-7c1f1b00-3506-11e9-8edb-fd6fe654ac12.gif)<br>

- エピソード：5000 回 / 迷路を解くのにかかったステップ数：5<br>
![rl_env_episode4999](https://user-images.githubusercontent.com/25688193/53063317-05cee880-3507-11e9-931e-8d4ced302785.gif)<br>

<br>

- 初回の行動方策 policy の値

|状態 s|行動 a0="Up"|行動 a1="Right"|行動 a2="Down"|行動 a3="Left"|
|---|---|---|---|---|
|s0|0.|0.5|0.5|0.|
|s1|0.|0.5|0.|0.5|
|s2|0.|0.|0.5|0.5|
|s3|0.33333333|0.33333333|0.33333333|0.|
|s4|0.|0.|0.5|0.5|
|s5|1.|0.|0.|0.|
|s6|1.|0.|0.|0.|
|s7|0.5|0.5|0.|0.|

- 方策勾配法による学習後の行動方策の値

|状態 s|行動 a0="Up"|行動 a1="Right"|行動 a2="Down"|行動 a3="Left"|
|---|---|---|---|---|
|s0|0.|0.01160017|0.98839983|0.|
|s1|0.|0.21923658|0.|0.78076342|
|s2|0.|0.|0.41565631|0.58434369|
|s3|0.00849945|0.98359742|0.00790313|0.|
|s4|0.|0.|0.98669085|0.01330915|
|s5|1.|0.|0.|0.|
|s6|1.|0.|0.|0.|
|s7|0.00996313|0.99003687|0.|0.|


### ◎ コードの内容説明
<!--
本コードの大まかな流れは、以下のようになる。<br>

### 1. エージェントの行動方策 `_policy` のためのパラメーター `_brain_parameters` を初期化する。
- この初期化処理は、`MazePolicyGradientBrain` クラスのコンストラクタとそのコンストラクタ内でコールされる `MazePolicyGradientBrain.init__brain_parameters()` メソッドにて行う。<br>
- パラメーター `_brain_parameters` の値は、行を状態 {s0,s1,s2,s3,s4,s5,s6,s7} 、列を行動 {"Up","Right","Down","Left"} とする表形式表現で実装する。（※行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。）<br>
- 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。<br>
- 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。<br>

```python
[MazePolicyGradientBrain.py]
class MazePolicyGradientBrain
    ...
    def __init__( self ):
        ...
        self._brain_parameters = self.init__brain_parameters()
        ...
        return

    def init__brain_parameters( self ):
        """
        方策パラメータを初期化
        """
        # 表形式（行：状態 s、列：行動 a）
        brain_parameters = np.array(
            [   # a0="Up", a1="Right", a3="Down", a4="Left"
                [ np.nan, 1,        1,         np.nan ], # s0
                [ np.nan, 1,        np.nan,    1 ],      # s1
                [ np.nan, np.nan,   1,         1 ],      # s2
                [ 1,      1,        1,         np.nan ], # s3
                [ np.nan, np.nan,   1,         1 ],      # s4
                [ 1,      np.nan,   np.nan,    np.nan ], # s5
                [ 1,      np.nan,   np.nan,    np.nan ], # s6
                [ 1,      1,        np.nan,    np.nan ], # s7
            ]
        )
        return brain_parameters
```

### 2. softmax 関数に従って、パラメーター `_brain_parameters` から行動方策 `_policy` を求める。
- この処理の初回処理（＝初回の行動方策の算出）は、`MazePolicyGradientBrain` クラスのコンストラクタからコールされる `convert_into_policy_from_brain_parameters()` メソッドにて行う。<br>
- それ以降のエピソードでの処理は、`MazePolicyGradientBrain` クラスの `decision_policy()` メソッドからコールされる `convert_into_policy_from_brain_parameters()` メソッドにて行う。<br>

```python
[MazePolicyGradientBrain.py]
class MazePolicyGradientBrain
    ...
    def __init__( self ):
        ...
        self._brain_parameters = self.init__brain_parameters()
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        return
    
    def convert_into_policy_from_brain_parameters( self, brain_parameters ):
        """
        方策パラメータから、行動方針 [policy] を決定する
        ・softmax 関数で確率を計算
        """
        beta = 1.0
        [m, n] = brain_parameters.shape
        policy = np.zeros( shape = (m,n) )

        theta = brain_parameters
        exp_theta = np.exp( beta * theta )

        for i in range(0, m):
            # 割合の計算
            policy[i, :] = exp_theta[i, :] / np.nansum( exp_theta[i, :] )

        # NAN 値は 0 に変換
        policy = np.nan_to_num( policy )

        return policy

    def decision_policy( self ):
        """
        行動方針を決定する
        """
        ...

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return self._policy
```

- ここで、初期化後の行動方策 `_policy` の値は、以下のような値になる。<br>
    ```python
    _policy : 
    [[ 0.          0.5         0.5         0.        ]
    [ 0.          0.5         0.          0.5       ]
    [ 0.          0.          0.5         0.5       ]
    [ 0.33333333  0.33333333  0.33333333  0.        ]
    [ 0.          0.          0.5         0.5       ]
    [ 1.          0.          0.          0.        ]
    [ 1.          0.          0.          0.        ]
    [ 0.5         0.5         0.          0.        ]]
    ```


### 3. 以下の 3.1 ~ 3.2 の処理を、エピソード `episode` 毎に繰り返す。

#### 3-1. エージェントをゴールまで移動させる。
- この処理は、`Academy` クラスからコールバックされる `MazeAgent` クラスのコールバック関数 `agent_action()` にて行う。
- まず、`agent_reset()` メソッドをコールし、エージェントの状態を初期位置にリセットする。
- エージェントの次行動 `next_action` は、確率で表現される行動方策 `policy` を元に、`np.random.choice()` メソッドで決定する。
- 迷路の各格子の位置を 0 ~ 8 の番号で管理しているので、これに従って、行動 {"Up","Right","Down","Left"} による次状態 `_state` =0 ~ 8 を指定する。<br>
    - 例えば、上に移動するときは次状態 `_state` の数字が3小さくなるなど。
- この際、エージェントの状態の履歴 `_states_history` と行動の履歴 `_action_history` を保管しておく。（後述の方策勾配法に基づく行動方策のためのパラメーターの更新処理で利用するため）
- while ループで、エージェントがゴールに辿り着くまで、これらのエージェントの移動処理を繰り返す。

```python
[Academy.py]
class Academy( object ):
    ...
    def academy_step( self ):
        """
        エピソードを１ステップ間隔実行する。
        """
        for episode in range( 0,self._max_episode ):
            for agent in self._agents:
                agent.agent_step( episode )
                agent.agent_action( episode )

                # 全ての Agent が完了時に break するように要修正
                if ( agent.IsDone() == True ):
                    break
                
            if ( agent.IsDone() == True ):
                break

        # Academy と全 Agents のエピソードを完了
        self._done = True
        for agent in self._agents:
            agent.agent_on_done()

        return
```

```python
[MazeAgent.py]
class MazeAgent
    ...
    def agent_action( self, episode ) :
        """
        各エピソードでのエージェントのアクションを記述
        ・Academy からコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
        """
        ...
        print( "現在のエピソード数：", episode )

        policy = self._brain.get_policy()

        #------------------------------------------------------------
        # 行動方策に基づき、エージェントを迷路のゴールまで移動させる。
        #------------------------------------------------------------
        # エージェントの状態を再初期化して、開始位置に設定する。
        self.agent_reset()

        # Goal にたどり着くまでループ
        while(1):
            # Brain のロジックに従ったエージェント次の行動
            next_action = self._brain.next_action( state = self._state )
            
            # エージェントの移動
            if next_action == "Up":
                self._state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
                action = 0
            elif next_action == "Right":
                self._state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
                action = 1
            elif next_action == "Down":
                self._state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
                action = 2
            elif next_action == "Left":
                self._state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる
                action = 3

            # 現在の状態の行動を設定
            self._action_history[-1] = action

            # 次の状態を追加
            self._states_history.append( self._state )
            self._action_history.append( np.nan )       # 次の状態での行動はまだ分からないので NaN 値を入れておく。

            # ゴールの指定
            if( self._state == 8 ):
                self.add_reword( 1.0 )  # ゴール地点なら、報酬
                break                       

        print( "迷路を解くのにかかったステップ数：" + str( len(self._states_history) ) )
        ...
```

#### 3.2. エージェントのゴールまでの履歴を元に、方策勾配法に従って、行動方策を更新する。

```python
[MazeAgent.py]
class MazeAgent
    ...
    def agent_action( self, episode ) :
        """
        各エピソードでのエージェントのアクションを記述
        ・Academy からコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
        """
        ...
        #------------------------------------------------------------
        # エージェントのゴールまでの履歴を元に、行動方策を更新
        #------------------------------------------------------------
        new_policy = self._brain.decision_policy()
        ...
```

### 4. 前回の行動方針との差分が十分小さければ、学習が完了したとみなし、エピソードを完了する。

```python
[MazeAgent.py]
class MazeAgent
    ...
    def agent_action( self, episode ) :
        """
        各エピソードでのエージェントのアクションを記述
        ・Academy からコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
        """
        done = False            # エピソードの完了フラグ
        stop_epsilon = 0.001    # エピソードの完了のための行動方策の差分値
        ...
        #------------------------------------------------------------
        # エピソードの完了判定処理
        #------------------------------------------------------------
        # 前回の行動方針との差分が十分小さくなれば学習を終了する。
        delta_policy = np.sum( np.abs( new_policy - policy ) )
        print( "前回の行動方針との差分：", delta_policy )

        if( delta_policy < stop_epsilon ):
            done = True

        return
```
-->