# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/24] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Brain import Brain
from Agent import Agent


class MazeQlearningBrain( Brain ):
    """
    迷宮問題の Brain。
    ・Qlearning による迷路検索用のアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _brain_parameters : list <float> / shape = [n_states, n_actions]
                行動方策 π を決定するためのパラメーター Θ
                ※ 行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。
                ※ 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。
                ※ 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。
        _policy : list<float> / shape = [n_states, n_actions]
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
        brain_parameters,
        epsilon = 0.5, gamma = 0.9, learning_rate = 0.1 
    ):
        super().__init__( n_states, n_actions )
        self._brain_parameters = brain_parameters
        self._policy = np.zeros( shape = ( self._n_states, self._n_actions ) )
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        self._q_function = self.init_q_function( brain_parameters = self._brain_parameters )
        self._epsilon = epsilon
        self._gamma = gamma
        self._learning_rate = learning_rate
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeQlearningBrain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "_policy : \n", self._policy )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "_q_function : \n", self._q_function )
        print( "_epsilon : \n", self._epsilon )
        print( "_gamma : \n", self._gamma )
        print( "_learning_rate : \n", self._learning_rate )
        print( "----------------------------------" )
        return

    def get_q_function( self ):
        """
        Q 関数の値を取得する。
        """
        return self._q_function

    def decay_learning_rate( self ):
        """
        学習率を減衰させる。
        """
        self._learning_rate = self._learning_rate / 2.0
        return

    def decay_epsilon( self ):
        """
        ε-greedy 法の ε 値を減衰させる。
        """
        self._epsilon = self._epsilon / 2.0
        return

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
        ・ε-グリーディー法に従った行動選択
        [Args]
            state : <int> 現在の状態
        """
        # ε-グリーディー法に従った行動選択
        if( self._epsilon <= np.random.rand() ):
            # ε の確率でランダムな行動を選択
            action = np.random.choice( self._n_actions, p = self._policy[ state, : ] )
            """
            action = np.random.choice( 
                self._actions,                  # アクションのリストから抽出
                p = self._policy[ state, : ]    # 抽出は、policy の確率に従う
            )
            """

        else:
            # Q の最大化する行動を選択
            action = np.nanargmax( self._q_function[state, :] )
            #action = self._actions[ np.nanargmax( self._q_function[state, :] ) ]

        if( action == np.nan ):
            action = 0

        return action


    def init_q_function( self, brain_parameters ):
        """
        Q 関数を表形式で初期化する。
        """
        # エピソードの開始時点では、最適な Q 関数の値は不明なので、ランダム値で初期化
        # brain_parameters をかけることで、壁方向は np.nan に設定する。
        [a, b] = brain_parameters.shape
        q_function = np.random.rand( a, b ) * brain_parameters
        return q_function


    def update_q_function( self, state, action, next_state, next_action, reword ):
        """
        Q 関数の値を更新する。

        [Args]
            state : <int> 現在の状態 s
            action : <int> 現在の行動 a
            next_state : <int> 次の状態 s'
            next_action : <int> 次の行動 a'
            reword : <float> 報酬
        
        [Returns]

        """
        # ゴールした場合
        if( next_state == 8 ):
            self._q_function[ state, action ] += self._learning_rate * ( reword - self._q_function[ state, action ] )
        else:
            # Qlearning : self._gamma * np.nanmax( self._q_function[ next_state, : ] )
            # Sarsa : self._gamma * self._q_function[ next_state, action ]
            self._q_function[ state, action ] += self._learning_rate * ( reword + self._gamma * np.nanmax( self._q_function[ next_state, : ] ) - self._q_function[ state, action ] )

        return self._q_function
