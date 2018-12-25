# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/07] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Brain import Brain


class MazeRamdomBrain( Brain ):
    """
    迷宮問題の Brain
    ・経路選択ロジックは、等確率な方向から１つを無作為選択
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _brain_parameters : 動的な型
                行動方策 π を決定するためのパラメーター

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        states = [ "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8" ],
        actions = [ "Up", "Right", "Down", "Left" ]
    ):
        super().__init__( states, actions )
        self._states = states
        self._actions = actions
        self._policy = np.zeros(
            shape = ( len(self._states), len(self._actions) ) 
        )
        self._brain_parameters = self.init_brain_parameters()
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeRamdomBrain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_states : \n", self._states )
        print( "_actions : \n", self._actions )
        print( "_policy : \n", self._policy )
        print( "_observations : \n", self._observations )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "----------------------------------" )
        return

    def reset_brain( self ):
        """
        Brain を再初期化する。
        """
        self._policy = np.zeros(
            shape = ( len(self._states), len(self._actions) ) 
        )
        self._brain_parameters = self.init_brain_parameters()
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
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


    def next_action( self, state ):
        """
        Brain のロジックに従って、次の行動を決定する。
        [Args]
            state : int
                現在の状態
        """
        # 行動方策 policy の確率に従って、次の行動を選択
        next_action = np.random.choice( 
            self._actions,                  # アクションのリストから抽出
            p = self._policy[ state, : ]    # 抽出は、policy の確率に従う
        )

        return next_action


    def decision_policy( self ):
        """
        行動方針を決定する
        """
        # エージェントの状態を取得
        self._observations = self._agent.collect_observations()

        # 行動の方策のためのパラメーターを更新

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return self._policy