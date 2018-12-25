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


class MazePolicyGradientBrain( Brain ):
    """
    迷宮問題の Brain。
    ・方策勾配法による迷路検索用のアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _brain_parameters : list< int, int >
                行動方策 π を決定するためのパラメーター Θ
                ※ 行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。
                ※ 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。
                ※ 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。

        _learning_rate : float
                学習率

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        states,
        actions,
        brain_parameters,
        learning_rate = 0.1
    ):
        super().__init__( states, actions )
        self._states = states
        self._actions = actions
        self._policy = np.zeros(
            shape = ( len(self._states) - 1, len(self._actions) ) 
        )
        self._brain_parameters = brain_parameters
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )
        self._learning_rate = learning_rate
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazePolicyGradientBrain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_states : \n", self._states )
        print( "_actions : \n", self._actions )
        print( "_policy : \n", self._policy )
        print( "_observations : \n", self._observations )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "_learning_rate : \n", self._learning_rate )
        print( "----------------------------------" )
        return


    def update_brain_parameter( self, brain_parameters, s_a_historys, policy ):
        """
        方策勾配法に従って、行動方策のためのパラメーターを更新する。
        Θ(s_i,a_j) = Θ(s_i,a_j) + η * ΔΘ(s,a_j)
        ΔΘ(s,a_j) = { N(s_i,a_j) + P(s_i,a_j)*N(s_i,a) } / n_steps

        [Args]
            brain_parameters : ndarry / shape =[m,n]
                更新前の行動方策のためのパラメーター
            s_a_historys : list<state>
                エージェントの状態と行動の履歴 [ [0,Down],[1,Up],...]
            learning_rate : float
                学習率

        """
        n_steps = len( s_a_historys ) - 1 # ゴールまでの総ステップ数
        if( n_steps == 0):
            n_steps = 1

        [m,n] = brain_parameters.shape

        #--------------------------------------------------------------
        # ΔΘ(s,a_j) = { N(s_i,a_j) + P(s_i,a_j)*N(s_i,a) } / n_steps
        #--------------------------------------------------------------
        # 参照コピーではなく、ディープコピー
        delta_brain_parameters = brain_parameters.copy()

        # ΔΘ(s,a_j) を要素 (s,a_j) 毎に定める。
        for i in range(0,m):
            for j in range(0,n):
                # thetaがnanでない場合
                if( np.isnan( brain_parameters[i,j] ) == False ):
                    # エージェントの状態履歴から、状態 i のもののみを取り出す
                    SA_i = [SA for SA in s_a_historys if SA[0] == i]

                    # 
                    SA_ij = [ SA for SA in s_a_historys if SA == [i, j] ]

                    #
                    N_i = len(SA_i)    # 状態iで行動した総回数
                    N_ij = len(SA_ij)  # 状態iで行動jをとった回数

                    # 
                    delta_brain_parameters[i, j] = (N_ij - policy[i, j] * N_i) / n_steps

        #-------------------------------------------------------
        # Θ(s_i,a_j) = Θ(s_i,a_j) + η * ΔΘ(s,a_j)
        #-------------------------------------------------------
        new_brain_parameters = brain_parameters + self._learning_rate * delta_brain_parameters

        return new_brain_parameters


    def convert_into_policy_from_brain_parameters( self, brain_parameters ):
        """
        方策パラメータから、行動方針 [policy] を決定する
        ・softmax 関数で確率を計算
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
        self._brain_parameters = self.update_brain_parameter(
            brain_parameters = self._brain_parameters,
            s_a_historys = self._observations[1],
            policy = self._policy
        )

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return self._policy
