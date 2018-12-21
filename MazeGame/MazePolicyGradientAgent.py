
# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/04] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# 自作クラス
from Agent import Agent
from MazeAgent import MazeAgent

class MazePolicyGradientAgent( MazeAgent ):
    """
    """
    def __init__( self, brain = None ):
        return super().__init__( brain )

    def goal_maze( self, policy ):
        """
        行動方策に基づき、迷宮のゴール地点までエージェントを移動
        """
        self.agent_reset()

        #------------------------------------------------
        # Goal にたどり着くまでループ
        #------------------------------------------------
        while(1):
            #------------------------------------------------
            # 行動方針の確率に従った次のエージェントの action
            #------------------------------------------------
            action = self._brain.action()
            next_action = np.random.choice( 
                action,
                p = policy[ self._state, : ]
            )

            # エージェントの移動
            if next_action == "Up":
                self._state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
                self._action = 0
            elif next_action == "Right":
                self._state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
                self._action = 1
            elif next_action == "Down":
                self._state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
                self._action = 2
            elif next_action == "Left":
                self._state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる
                self._action = 3

            # 現在の状態の行動を設定
            self._action_history[-1] = self._action

            # 次の状態を追加
            self._states_history.append( self._state )
            self._action_history.append( np.nan )       # 次の状態での行動はまだ分からないので NaN 値を入れておく。

            #------------------------------------------------
            # ゴールの指定
            #------------------------------------------------
            # ゴール地点なら、報酬
            if( self._state == 8 ):
                self.add_reword( 1.0 )
                break

        return


    def agent_action( self, step ) :
        """
        各エピソードでのエージェントのアクションを記述
        ・Academy からコールされるコールバック関数
        """
        done = False
        stop_epsilon = 0.001
        #self._brain.reset_brain()
        print( "エピソードのステップ数：", step )

        policy = self._brain.get_policy()

        # 行動方策に基づき、エージェントを迷路のゴールまで移動させる。
        self.goal_maze( policy )
        print( "迷路を解くのにかかったステップ数：" + str( len(self._states_history) ) )

        # エージェントのゴールまでの履歴を元に、行動方策を更新
        new_policy = self._brain.decision_policy()

        # 前回の行動方針との差分が十分小さくなれば学習を終了する。
        delta_policy = np.sum( np.abs( new_policy - policy ) )
        print( "前回の行動方針との差分：", delta_policy )

        if( delta_policy < stop_epsilon ):
            done = True

        return