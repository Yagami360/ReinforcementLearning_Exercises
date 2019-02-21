# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/02/20] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Brain import Brain


class MazeEveryVisitMCBrain( Brain ):
    """
    迷宮問題の Brain
    ・経路選択ロジックは、逐一訪問モンテカルロ法
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
        epsilon = 0.5, gamma = 0.9 
    ):
        super().__init__( n_states, n_actions )
        self._brain_parameters = brain_parameters
        self._policy = np.zeros( shape = ( self._n_states, self._n_actions ) )
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        self._q_function = self.init_q_function( brain_parameters = self._brain_parameters )
        self._epsilon = epsilon
        self._gamma = gamma
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
        print( "_q_function : \n", self._q_function )
        print( "_epsilon : \n", self._epsilon )
        print( "_gamma : \n", self._gamma )
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
        self._epsilon = self._epsilon / 1.10
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
        rand = np.random.rand()
        if( self._epsilon >= rand ):
            # ε の確率でランダムな行動を選択
            action = np.random.choice( self._n_actions, p = self._policy[ state, : ] )

        else:
            # Q の最大化する行動を選択
            action = np.nanargmax( self._q_function[state, :] )

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


    def update_q_function( self, s_a_r_historys ):
        """
        Brain を更新する。
        [Args]
            s_a_r_historys : <list> １回のエピソードでの (s,a,r) の履歴
            total_reward : <float> 割引利得の総和
        """
        from collections import defaultdict
        N = defaultdict( lambda: [0] * self._n_actions )

        total_reward = 0.0  # 状態行動対 (s,a) からみた将来の報酬和

        # 逐次訪問MC法による方策評価
        # t = 0 ~ T
        for (t, s_a_r) in enumerate(s_a_r_historys):
            state = s_a_r[0]
            action = s_a_r[1]
            reward = s_a_r[2]

            # i = t ~ T / 0 ~ T, 1 ~ T, 2 ~ T,...
            k = 0
            for i in range( t, len(s_a_r_historys) ):
                # reward_toal = Σ_k=0^T gamma^k * r_{t+k+1}
                total_reward += ( self._gamma ** k ) * s_a_r_historys[i][2]
                k += 1

            # 状態行動対 (s,a) に遷移した回数
            N[state][action] += 1

            # 学習率（平均化のため逆数）
            alpha = 1 / N[state][action]

            # ゴール状態での行動価値関数は定義していないため、除外
            if( state != 8 ):
                self._q_function[state][action] += alpha * ( total_reward - self._q_function[state][action] )

        return