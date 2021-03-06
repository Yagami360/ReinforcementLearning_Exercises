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
        _brain_parameters : list <float> / shape = [n_states, n_actions]
                行動方策 π を決定するためのパラメーター Θ
                ※ 行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。
                ※ 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。
                ※ 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。
        _policy : list<float> / shape = [n_states, n_actions]

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        n_states,
        n_actions,
        brain_parameters,
    ):
        super().__init__( n_states, n_actions )
        self._brain_parameters = brain_parameters
        self._policy = np.zeros( shape = ( self._n_states, self._n_actions ) )
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeRamdomBrain" )
        print( self )
        print( str )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "_policy : \n", self._policy )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "----------------------------------" )
        return

    def get_policy( self ):
        return self._policy

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


    def action( self, state ):
        """
        Brain のロジックに従って、現在の状態 s での行動 a を決定する。
        [Args]
            state : int
                現在の状態
        """
        # 行動方策 policy の確率に従って、状態 s での行動 a を選択
        action = np.random.choice( 
            self._n_actions, 
            p = self._policy[ state, : ]    # 抽出は、policy の確率に従う
        )

        return action


    def update( self ):
        """
        Brain を更新する。
        """
        # エージェントの状態を取得

        # 行動の方策のためのパラメーターを更新

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return