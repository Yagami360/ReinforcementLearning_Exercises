# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/02/18] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Brain import Brain


class FrozenLakeRamdomBrain( Brain ):
    """
    FrozenLake の Brain
    ・経路選択ロジックは、等確率な方向から１つを無作為選択
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        n_states,
        n_actions
    ):
        super().__init__( n_states, n_actions )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "FrozenLakeRamdomBrain" )
        print( self )
        print( str )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "----------------------------------" )
        return

    def action( self, state ):
        """
        Brain のロジックに従って、現在の状態 s での行動 a を決定する。
        [Args]
            state : int
                現在の状態
        """
        # 行動方策 policy の確率に従って、状態 s での行動 a を選択
        action = np.random.randint( self._n_actions )

        return action