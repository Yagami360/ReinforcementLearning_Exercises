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
from MazeAgent import MazeAgent


class MazeSarsaBrain( Brain ):
    """
    迷宮問題の Brain。
    ・Sarsa による迷路検索用のアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _brain_parameters : list< int, int >
                行動方策 π を決定するためのパラメーター Θ
                ※ 行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。
                ※ 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。
                ※ 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。

        _q_function : 行動状態関数 Q(s,a)
                      行を状態 s, 列を行動 a とする表形式表現

        _epsilon : float
                   ε-greedy 法の ε 値
        _gamma : float
                割引利得の γ 値
        _learning_rate : float
                学習率

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        states,
        actions,
        brain_parameters,
        epsilon = 0.5, gamma = 0.9, learning_rate = 0.1 
    ):
        super().__init__( states, actions )
        self._states = states
        self._actions = actions
        self._brain_parameters = brain_parameters
        self._policy = np.zeros(
            shape = ( len(self._states), len(self._actions) )
        )
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        self._q_function = self.init_q_function( brain_parameters = self._brain_parameters )
        self._epsilon = epsilon
        self._gamma = gamma
        self._learning_rate = learning_rate
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeSarsaBrain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_states : \n", self._states )
        print( "_actions : \n", self._actions )
        print( "_policy : \n", self._policy )
        print( "_observations : \n", self._observations )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "_q_function : \n", self._q_function )
        print( "_epsilon : \n", self._epsilon )
        print( "_gamma : \n", self._gamma )
        print( "_learning_rate : \n", self._learning_rate )
        print( "----------------------------------" )
        return


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
            state : int
                現在の状態 s
            action : str
                現在の行動 a
            next_state : int
                次の状態 s'
            next_action : str
                次の行動 a'
            reword : float
                報酬
        
        [Returns]

        """
        # ゴールした場合
        if( next_state == 8):
            self._q_function[ state, action ] += self._learning_rate * ( reword - self._q_function[ state, action ] )
        else:
            self._q_function[ state, action ] += self._learning_rate * ( reword + self._gamma * self._q_function[ next_state, next_action ] - self._q_function[ state, action ] )

        return self._q_function


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
                self._actions,                  # アクションのリストから抽出
                p = self._policy[ state, : ]    # 抽出は、policy の確率に従う
            )

        else:
            # Q の最大化する行動を選択
            next_action = self._actions[ np.nanargmax( self._q_function[state, :] ) ]

        # このメソッドが呼び出される度に、ε の値を徐々に小さくする。
        self._epsilon = self._epsilon / 2

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


    def decision_policy( self ):
        """
        行動方針を決定する
        """
        # エージェントの状態を取得
        self._observations = self._agent.collect_observations()
        state = self._observations[0]
        s_a_historys = self._observations[1]

        # 行動の方策のためのパラメーターを更新
        """
        self.update_q_function(
            state = state,
            action = self._action
        )
        """

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return self._policy
