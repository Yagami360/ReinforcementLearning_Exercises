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
        _policy : float
            行動方策 π。確率値（0~1）
        _theta : 
            方策を決定するためのパラメータ

        _state : int
            エージェントの状態 s
        _states_history : list <state>
            エージェントの状態の履歴

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self, brain = None ):
        super().__init__( brain )
        self._policy = 0.0
        self._theta = []
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
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
        print( "_policy : \n", self._policy )
        print( "_theta : \n", self._theta )        
        print( "----------------------------------" )
        return

    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        self.add_vector_obs( self._state )
        self.add_vector_obs( self._states_history )
        return


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        super().agent_reset()
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        self._policy = 0.0
        self.init_theta()
        self.convert_into_policy_from_theta( self._theta )
        return
        
    def init_theta( self ):
        """
        方策パラメータを初期化
        """
        self._theta = np.array(
            [
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
        return self._theta

    def convert_into_policy_from_theta( self, theta ):
        """
        方策パラメータから、行動方針 [policy] を決定する
        """
        [m, n] = theta.shape
        self._policy = np.zeros( shape = (m,n) )
        for i in range(0, m):
            # 割合の計算
            self._policy[i, :] = theta[i, :] / np.nansum( theta[i, :] )

        # NAN 値は 0 に変換
        self._policy = np.nan_to_num( self._policy )

        return self._policy

    def state_history( self ):
        return self._states_history


    def agent_action( self, step ):
        """
        現在の状態に基づき、エージェントの実際のアクションを記述する。

        [Input]

        """
        super().agent_action( step )
        done = False
        self._states_history.append( self._state )

        action = self._brain.action()

        #-------------------------------------------
        # エージェントの意思決定ロジック
        # → Brain で処理したい
        #-------------------------------------------
        # 行動方針の確率に従って、action が選択される
        next_action = np.random.choice( 
            action,
            p = self._policy[ self._state, : ]
        )

        #-------------------------------------------
        # エージェントの移動
        #-------------------------------------------
        if next_action == "Up":
            self._state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
        elif next_action == "Down":
            self._state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
        elif next_action == "Left":
            self._state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる
        elif next_action == "Right":
            self._state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
        
        #-------------------------------------------
        # 報酬の指定
        #-------------------------------------------
        #self.add_reword( 0.1 )

        #-------------------------------------------
        # エピソードの完了
        #-------------------------------------------
        # ゴール地点なら、完了フラグ ON
        if( self._state == 8 ):
            self.Done()
            self._states_history.append( self._state )
            
        return

