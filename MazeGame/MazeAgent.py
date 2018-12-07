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

        _action : int
            上下移動のアクション
        _action_istory : list<action>
            アクションの履歴

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, brain = None ):
        super().__init__( brain )
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        self._action = None
        self._action_history = []
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
        print( "_action : \n", self._action )
        print( "_action_history : \n", self._action_history )
        print( "----------------------------------" )
        return
    
    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        self.add_vector_obs( self._state )
        self.add_vector_obs( self._states_history )
        self.add_vector_obs( self._action )
        self.add_vector_obs( self._action_history )
        return


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        super().agent_reset()
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        self._action = None
        self._action_history = []
        return

    def state_history( self ):
        return self._states_history

    def agent_action( self, step ):
        """
        現在の状態に基づき、エージェントの実際のアクションを記述する。

        [Input]

        """
        super().agent_action( step )
        done = False

        #------------------------------------------------
        # エージェントの意思決定ロジック
        #------------------------------------------------
        policy = self._brain.decision_policy()

        #------------------------------------------------
        # 行動方針の確率に従った次のエージェントの action
        #------------------------------------------------
        action = self._brain.action()
        #np.random.seed(8)
        next_action = np.random.choice( 
            action,
            p = policy[ self._state, : ]
        )
        
        # エージェントの移動
        if next_action == "Up":
            self._state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
            self._action = 0
        elif next_action == "Right":
            self._state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
            self._action = 1
        elif next_action == "Down":
            self._state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
            self._action = 2
        elif next_action == "Left":
            self._state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる
            self._action = 3

        self._states_history.append( self._state )
        self._action_history.append( self._action )

        #------------------------------------------------
        # 報酬の指定
        #------------------------------------------------
        # ゴール地点なら、報酬
        if( self._state == 8 ):
            self.add_reword( 1.0 )

        #------------------------------------------------
        # エピソードの完了
        #------------------------------------------------
        # ゴール地点なら、完了フラグ ON
        if( self._state == 8 ):
            self.Done()
            self._states_history.append( self._state )
            
        return

