# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/08] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Brain import Brain


class MazePolicyGradientBrain( Brain ):
    """
    迷宮問題の Brain。
    ・方策勾配法による迷路検索用のアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self ):
        super().__init__()
        self._action = ["Up", "Right", "Down", "Left"]
        self._policy = 0.0
        self._brain_parameters = self.init__brain_parameters()
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazePolicyGradientBrain" )
        print( self )
        print( str )
        
        print( "_action : \n", self._action )
        print( "_policy : \n", self._policy )
        print( "_observation : \n", self._observation )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "----------------------------------" )
        return

    def init__brain_parameters( self ):
        """
        方策パラメータを初期化
        """
        _brain_parameters = np.array(
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
        return _brain_parameters

    def convert_into_policy_from_brain_parameters( self, brain_parameters ):
        """
        方策パラメータから、行動方針 [policy] を決定する
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
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return self._policy
