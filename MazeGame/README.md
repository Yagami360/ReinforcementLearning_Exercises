# 強化学習のトイプロブレム用の迷路探索問題
> 実装中...

強化学習の学習環境用としての迷路探索問題。<br>
単純な迷路探索問題を、Unity ML-Agents のフレームワーク（`Academy`,`Brain`,`Agent`クラス など）を参考にして実装しています。<br>
分かりやすいように `main.py` ファイル毎に１つの完結した実行コードにしています。<br>

- 【参考文献＆サイト】<br>
    - [Reinforcement Learning in Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Design.md)<br>
    - [Unityで強化学習していたAgentのソースコードを読む話](https://ensekitt.hatenablog.com/entry/2017/12/27/123000)<br> 


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [コード説明＆実行結果](#コード説明＆実行結果)
    1. [等確率による迷路検索問題 : `main1.py`](#コード説明＆実行結果１)
    1. [方策勾配法による迷路検索問題 : `main2.py`](#コード説明＆実行結果２)
    1. [Sarsaによる迷路検索問題 : `main3.py`](#コード説明＆実行結果３)
    1. Q学習による迷路検索問題 : `main4.py`
    1. モンテカルロ法による迷路検索問題 : `main5.py`
1. 背景理論
    1. [【外部リンク】強化学習 / 方策反復法](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#%E6%96%B9%E7%AD%96%E5%8F%8D%E5%BE%A9%E6%B3%95)
    1. [【外部リンク】強化学習 / モンテカルロ法による価値推定](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#%E3%83%A2%E3%83%B3%E3%83%86%E3%82%AB%E3%83%AB%E3%83%AD%E6%B3%95%E3%81%AB%E3%82%88%E3%82%8B%E4%BE%A1%E5%80%A4%E6%8E%A8%E5%AE%9A)
    1. [【外部リンク】強化学習 / Sarsa](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Sarsa)
    1. [【外部リンク】強化学習 / Q学習](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Q%E5%AD%A6%E7%BF%92)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1


<!--
<a id="使用するライブラリ"></a>

## ■ 使用するライブラリ


<a id="使用するデータセット"></a>

## ■ 使用するデータセット

-->

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

<a id="コード説明＆実行結果１"></a>

### ◎ 等確率による迷路検索問題 : `main1.py`
強化学習の学習環境用の迷路探索問題。<br>
等確率で表現された４つの移動候補（上下左右）から１つの方向を無作為に選択し、これを繰り返すことで、最終的に目的地に到着させる。<br>

![mazegame_random1](https://user-images.githubusercontent.com/25688193/49664161-67e58900-fa94-11e8-8df4-262c46446ddb.gif)<br>


<a id="コード説明＆実行結果２"></a>

### ◎ 方策勾配法による迷路検索問題 : `main2.py`
方策反復法の具体例なアルゴリズムの１つである方策勾配法によって、迷路探索問題を解く。<br>

#### ☆ コードの実行結果

- 以下のアニメーションは、方策勾配法によって、学習した行動方策 π に基づく迷路探索問題の探索結果である。<br>
    ![mazegame_policygradient1](https://user-images.githubusercontent.com/25688193/50348392-f4f00e00-057b-11e9-805a-8ee3a84b26f2.gif)<br>
    > 先の等確率による迷路検索問題 `main1.py` とは異なり、最短ルートで、ゴールまで到達できるようになっていることが分かる。<br>

- xxx
```python
エピソードのステップ数： 1
迷路を解くのにかかったステップ数：23
前回の行動方針との差分： 0.018181646008
エピソードのステップ数： 2
迷路を解くのにかかったステップ数：13
前回の行動方針との差分： 0.0234385255449
エピソードのステップ数： 3
迷路を解くのにかかったステップ数：7
前回の行動方針との差分： 0.0323087713502
エピソードのステップ数： 4
迷路を解くのにかかったステップ数：7
前回の行動方針との差分： 0.0319429956863
エピソードのステップ数： 5
迷路を解くのにかかったステップ数：15
前回の行動方針との差分： 0.0185799607072
エピソードのステップ数： 6
迷路を解くのにかかったステップ数：77
前回の行動方針との差分： 0.00904734008632
エピソードのステップ数： 7
迷路を解くのにかかったステップ数：81
前回の行動方針との差分： 0.00934323007014
エピソードのステップ数： 8
迷路を解くのにかかったステップ数：97
前回の行動方針との差分： 0.00924493760322
エピソードのステップ数： 9
迷路を解くのにかかったステップ数：43
前回の行動方針との差分： 0.0104009444956
エピソードのステップ数： 10
迷路を解くのにかかったステップ数：19
前回の行動方針との差分： 0.0205719688879
...
エピソードのステップ数： 4995
迷路を解くのにかかったステップ数：5
前回の行動方針との差分： 6.85450559647e-05
エピソードのステップ数： 4996
迷路を解くのにかかったステップ数：5
前回の行動方針との差分： 6.84591462685e-05
エピソードのステップ数： 4997
迷路を解くのにかかったステップ数：5
前回の行動方針との差分： 6.83733982803e-05
エピソードのステップ数： 4998
迷路を解くのにかかったステップ数：5
前回の行動方針との差分： 6.82878115942e-05
エピソードのステップ数： 4999
迷路を解くのにかかったステップ数：5
前回の行動方針との差分： 6.82023858042e-05
```

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
|s0|0.|0.01222508|0.98777492|0.|
|s1|0.|0.28989193|0.|0.71010807|
|s2|0.|0.|0.40313477|0.59686523|
|s3|0.01068343|0.9813875|0.00792907|0.|
|s4|0.|0.|0.98764577|0.01235423|
|s5|1.|0.|0.|0.|
|s6|1.|0.|0.|0.|
|s7|0.01129659|0.98870341|0.|0.|


#### ☆ コードの内容説明
本コードの大まかな流れは、以下のようになる。<br>

#### 1. エージェントの行動方策 `_policy` のためのパラメーター `_brain_parameters` を初期化する。
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

#### 2. softmax 関数に従って、パラメーター `_brain_parameters` から行動方策 `_policy` を求める。
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

#### 3. エージェントをゴールまで移動させる。
- この処理は、`Academy` クラスからコールバックされる `MazeAgent` クラスのコールバック関数 `agent_action()` にて行う。
- エージェントの次行動 `next_action` は、確率で表現される行動方策 `policy` を元に、`np.random.choice()` メソッドで決定する。
- 迷路の各格子の位置を 0~8 の番号で管理しているので、これに従って、行動 {"Up","Right","Down","Left"} による次状態 `_state` =0~8 を指定する。<br>
    - 例えば、上に移動するときは次状態 `_state` の数字が3小さくなるなど。
- この際、エージェントの状態の履歴 `_states_history` と行動の履歴 `_action_history` を保管しておく。（後述の方策勾配法に基づく行動方策のためのパラメーターの更新処理で利用するため）
- while ループで、エージェントがゴールに辿り着くまで、これらのエージェントの移動処理を繰り返す。

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
        [Returns]
            done : bool
                エピソードの完了フラグ
        """
        done = False            # エピソードの完了フラグ
        stop_epsilon = 0.001    # エピソードの完了のための行動方策の差分値

        #self._brain.reset_brain()
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

        #------------------------------------------------------------
        # エージェントのゴールまでの履歴を元に、行動方策を更新
        #------------------------------------------------------------
        new_policy = self._brain.decision_policy()

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



<a id="コード説明＆実行結果３"></a>

### ◎ Sarsaによる迷路検索問題 : `main3.py`

- 行動価値関数 Q(s,a) を初期化する。<br>
    - 行動価値関数 Q(s,a) は、行を状態 s、列を行動 a とする表形式で実装する。<br>
    - xxx
    ```python
    [SarsaBrain.py]
    ```
2. xxx

> 実装中...



---

<!--
<a id="背景理論"></a>

## 背景理論

<a id="背景理論１"></a>

## 背景理論１

-->

## デバッグメモ
```python
pi :
    array([ 0. ,  0.5,  0.5,  0. ])
    array([ 0. ,  0.5,  0. ,  0.5])
    array([ 0. ,  0. ,  0.5,  0.5])
    array([ 0.33333333,  0.33333333,  0.33333333,  0.        ])

theta :
    array([ nan,   1.,   1.,  nan])
    array([ nan,   1.,  nan,   1.])

delta_theta : N_i=38, N_ij=20, T=240
    array([        nan,  0.00416667,  1.        ,         nan])
    array([    nan,  0.0125,     nan, -0.0125])
    array([ nan,  nan,   0.,   0.])
    ...
    array([ 0.00625, -0.00625,      nan,      nan])


```