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
        print( "_agent : \n", self._agent )        
        print( "_action : \n", self._action )
        print( "_policy : \n", self._policy )
        print( "_observations : \n", self._observations )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "----------------------------------" )
        return

    def init__brain_parameters( self ):
        """
        方策パラメータを初期化
        """
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


    def update_brain_parameter( self, brain_parameters, state_action_historys, policy, learning_rate = 0.1 ):
        """
        方策勾配法に従って、行動方策のためのパラメーターを更新する。
        Θ(s_i,a_j) = Θ(s_i,a_j) + η * ΔΘ(s,a_j)
        ΔΘ(s,a_j) = { N(s_i,a_j) + P(s_i,a_j)*N(s_i,a) } / n_steps

        [Args]
            brain_parameters : ndarry / shape =[m,n]
                更新前の行動方策のためのパラメーター
            state_action_historys : list<state>
                エージェントの状態と行動の履歴 [ [0,Down],[1,Up],...]
            learning_rate : float
                学習率

        """
        n_steps = len( state_action_historys ) - 1 # ゴールまでの総ステップ数
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
                if( np.isnan( brain_parameters[i,j] ) != False ):
                    # エージェントの状態履歴から、状態 i のもののみを取り出す
                    SA_i = [SA for SA in state_action_historys if SA[0] == i]

                    # 
                    SA_ij = [ SA for SA in state_action_historys if SA == [i, j] ]

                    #
                    N_i = len(SA_i)    # 状態iで行動した総回数
                    N_ij = len(SA_ij)  # 状態iで行動jをとった回数

                    # 
                    delta_brain_parameters[i, j] = (N_ij - policy[i, j] * N_i) / n_steps

        #-------------------------------------------------------
        # Θ(s_i,a_j) = Θ(s_i,a_j) + η * ΔΘ(s,a_j)
        #-------------------------------------------------------
        new_brain_parameters = brain_parameters + learning_rate * delta_brain_parameters

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


    def decision_policy( self ):
        """
        行動方針を決定する
        """
        # エージェントの状態を取得
        self._observations = self._agent.collect_observations()
        
        state_action_historys = []
        for s,a in zip( self._observations[1], self._observations[3] ):
            state_action_historys.append( [s,a] )

        # 行動の方策のためのパラメーターを更新
        self._brain_parameters = self.update_brain_parameter(
            brain_parameters = self._brain_parameters,
            state_action_historys = state_action_historys,
            policy = self._policy,
            learning_rate = 0.1
        )

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。
        self._policy = self.convert_into_policy_from_brain_parameters( self._brain_parameters )

        return self._policy
