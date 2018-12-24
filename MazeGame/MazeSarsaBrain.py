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
from Agent import Agent
from MazeAgent import MazeAgent


class MazeSarsaBrain( Brain ):
    """
    迷宮問題の Brain。
    ・Sarsa による迷路検索用のアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _q_function : 行動状態関数 Q(s,a)
                      行を状態 s, 列を行動 a とする表形式表現

        _epsilon : float
                   ε-greedy 法の ε 値

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, epsilon = 0.5 ):
        super().__init__()
        self._action = ["Up", "Right", "Down", "Left"]
        self._policy = np.zeros( shape = (8, len(self._action)) )
        self._brain_parameters = self.init_brain_parameters()
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        self._q_function = self.init_q_function( brain_parameters = self._brain_parameters )
        self._epsilon = epsilon
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeSarsaBrain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )        
        print( "_action : \n", self._action )
        print( "_policy : \n", self._policy )
        print( "_observations : \n", self._observations )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "_q_function : \n", self._q_function )
        print( "_epsilon : \n", self._epsilon )
        print( "----------------------------------" )
        return

    def reset_brain( self ):
        """
        Brain を再初期化する。
        """
        self._policy = np.zeros( shape = (8, len(self._action)) )
        self._brain_parameters = self.init_brain_parameters()
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        self._q_function = self.init_q_function( brain_parameters = self._brain_parameters )
        return


    def init_brain_parameters( self ):
        """
        方策パラメータを初期化
        """
        # 表形式（行：状態 s、列：行動 a）
        # ※行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。
        # 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。
        # 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。
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


    def init_q_function( self, brain_parameters ):
        """
        Q 関数を表形式で初期化する。
        """
        # エピソードの開始時点では、最適な Q 関数の値は不明なので、ランダム値で初期化
        # brain_parameters をかけることで、壁方向は np.nan に設定する。
        [a, b] = brain_parameters.shape
        q_function = np.random.rand( a, b ) * brain_parameters
        return q_function


    def next_action( self, state ):
        """
        Brain のロジックに従って、次の行動を決定する。
        ・ε-グリーディー法に従った行動選択
        [Args]
            state : int
                現在の状態
        """
        # ε-グリーディー法に従った行動選択
        if( np.random.rand() < self._epsilon ):
            # ε の確率でランダムな行動を選択
            next_action = np.random.choice( 
                self._action,
                p = self._policy[ state, : ]
            )

        else:
            # Q の最大化する行動を選択
            next_action = self._action[ np.nanargmax( self._q_function[state, :] ) ]

        return next_action


    def convert_into_policy_from_brain_parameters( self, brain_parameters ):
        """
        方策パラメータから、行動方針 [policy] を決定する
        """
        [m, n] = brain_parameters.shape
        policy = np.zeros( shape = (m,n) )
        for i in range(0, m):
            # 割合の計算
            policy[i, :] = brain_parameters[i, :] / np.nansum( brain_parameters[i, :] )

        # NAN 値は 0 に変換
        policy = np.nan_to_num( policy )

        return policy