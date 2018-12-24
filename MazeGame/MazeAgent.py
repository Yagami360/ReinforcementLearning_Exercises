# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/04] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# 自作クラス
from Agent import Agent


class MazeAgent( Agent ):
    """
    迷路探索用エージェント。
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _state : int
            エージェントの状態 s
        _states_history : list <state>
            エージェントの状態の履歴

        _action_istory : list<action>
            アクションの履歴

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, brain = None ):
        super().__init__( brain )
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        self._action_history = []
        self._action_history.append( np.nan )
        self.agent_reset()
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeAgent" )
        print( self )
        print( str )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_state : \n", self._state )
        print( "_states_history : \n", self._states_history )
        print( "_action_history : \n", self._action_history )
        print( "----------------------------------" )
        return
    
    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        self._observations = []
        self.add_vector_obs( self._state )
        self.add_vector_obs( self._states_history )
        self.add_vector_obs( self._action_history )
        return self._observations


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        super().agent_reset()
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        self._action_history = []
        self._action_history.append( np.nan )
        return


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
