# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/04] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# 自作クラス
from AgentBase import AgentBase


class MazeAgent( AgentBase ):
    """
    迷路探索用エージェント。
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _policy : float
            行動方策 π。確率値（0~1）
        _theta : 
            方策を決定するためのパラメータ

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """

    def __init__( self ):
        super().__init__()
        """
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        self._policy = 0.0
        self._theta = []
        """
        self.agent_reset()
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "AgentBase" )
        print( self )
        print( str )
        
        print( "_state : \n", self._state )
        print( "_states_history : \n", self._states_history )
        print( "_policy : \n", self._policy )
        print( "_theta : \n", self._theta )        
        print( "----------------------------------" )
        return
    
    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
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

    def agent_step( self ):
        """
        エージェント [Agent] の次の状態を記述する。

        [Args]

        [Returns]
            next_state : Agent の次の状態
        """
        next_state = self._state
        return next_state
    
    def agent_action( self ):
        """
        現在の状態に基づき、エージェントの実際のアクションを記述する。

        [Input]

        """
        return

