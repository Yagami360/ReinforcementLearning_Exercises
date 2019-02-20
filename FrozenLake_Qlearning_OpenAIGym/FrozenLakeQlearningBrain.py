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


class FrozenLakeQlearningBrain( Brain ):
    """
    FrozenLake の Brain
    ・経路選択ロジックは、Q学習
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _q_function : list<float> / shape = [n_states, n_actions] 
                      行動状態関数 Q(s,a)
                      行を状態 s, 列を行動 a とする表形式表現
        _epsilon : <float> ε-greedy 法の ε 値
        _gamma : <float> 割引利得の γ 値
        _learning_rate : <float> 学習率

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        n_states,
        n_actions,
        epsilon = 0.5, gamma = 0.9, learning_rate = 0.1 
    ):
        super().__init__( n_states, n_actions )
        self._epsilon = epsilon
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._q_function = np.random.rand( n_states, n_actions )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "FrozenLakeQlearningBrain" )
        print( self )
        print( str )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "_epsilon : \n", self._epsilon )
        print( "_gamma : \n", self._gamma )
        print( "_learning_rate : \n", self._learning_rate )
        print( "_q_function : \n", self._q_function )
        print( "----------------------------------" )
        return

    def get_q_function( self ):
        """
        Q 関数の値を取得する。
        """
        return self._q_function

    def decay_epsilon( self ):
        """
        ε-greedy 法の ε 値を減衰させる。
        """
        self._epsilon = self._epsilon / 2.0
        return

    def action( self, state ):
        """
        Brain のロジックに従って、現在の状態 s での行動 a を決定する。
        [Args]
            state : int
                現在の状態
        """
        # ε-グリーディー法に従った行動選択
        if( self._epsilon >= np.random.rand() ):
            # ε の確率でランダムな行動を選択
            action = np.random.randint( self._n_actions )

        else:
            # Q の最大化する行動を選択
            action = np.nanargmax( self._q_function[state, :] )

        if( action == np.nan ):
            action = 0

        return action

    def update_q_function( self, state, action, next_state, reward ):
        """
        Q 関数の値を更新する。
        [Args]
            state : <int> 現在の状態 s
            action : <int> 現在の行動 a
            next_state : <int> 次の状態 s'
            reword : <float> 報酬
        
        [Returns]
        """
        # Qlearning : self._gamma * np.nanmax( self._q_function[ next_state, : ] )
        # Sarsa : self._gamma * self._q_function[ next_state, action ]
        self._q_function[ state, action ] += self._learning_rate * ( reward + self._gamma * np.nanmax( self._q_function[ next_state, : ] ) - self._q_function[ state, action ] )

        return self._q_function
