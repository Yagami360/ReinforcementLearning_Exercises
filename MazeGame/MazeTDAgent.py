# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/25] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# 自作クラス
from Agent import Agent
from MazeAgent import MazeAgent

class MazeTDAgent( MazeAgent ):
    """
    迷路探索用エージェント。
    ・迷宮探索の１ステップ毎に、行動方策を更新する。
    ・価値反復法（Sarsa, Q学習など）での使用を想定している。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, brain = None ):
        super().__init__( brain )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeTDAgent" )
        print( self )
        print( str )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_done : \n", self._done )
        print( "_state : \n", self._state )
        print( "_s_a_historys : \n", self._s_a_historys )
        print( "----------------------------------" )
        return


    def agent_action( self, episode ) :
        """
        各エピソードでのエージェントのアクションを記述
        ・Academy からコールされるコールバック関数
        ・迷路のスタートからゴールまでを、１エピソードとする。
        ・迷路検索の１ステップ毎に、行動方策を更新する。
        [Args]
            episode : 現在のエピソード数
        """
        done = False            # エピソードの完了フラグ
        stop_epsilon = 0.001    # エピソードの完了のための行動方策の差分値

        print( "現在のエピソード数：", episode )
        policy = self._brain.get_policy()

        #------------------------------------------------------------
        # 行動方策に基づき、エージェントを迷路のゴールまで移動させる。
        # ※ 迷路のスタートからゴールまでを、１エピソードとする。
        #------------------------------------------------------------
        # エージェントの状態を再初期化して、開始位置に設定する。
        self.agent_reset()

        # Goal にたどり着くまでループ
        while(1):
            # Brain のロジックに従ったエージェント次の行動
            next_action = self._brain.next_action( state = self._state )
            
            # エージェントの移動
            if next_action == "Up":
                self._state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
                action = 0
            elif next_action == "Right":
                self._state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
                action = 1
            elif next_action == "Down":
                self._state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
                action = 2
            elif next_action == "Left":
                self._state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる
                action = 3

            # 現在の状態 s の行動 a を設定
            self._s_a_historys[-1][1] = action  # -1 で末端に追加

            # 次の状態 s'と行動 a' を追加
            # 次の状態での行動はまだ分からないので NaN 値を入れておく。
            self._s_a_historys.append( [self._state, np.nan] )
            
            # Q 関数を更新
            """
            self._brain.update_q_function(
                state = self._s_a_historys[-2][0],
            )
            """

            # ゴールの指定
            if( self._state == 8 ):
                self.add_reword( 1.0 )  # ゴール地点なら、報酬
                break                       

        print( "迷路を解くのにかかったステップ数：" + str( len(self._s_a_historys) ) )

        #------------------------------------------------------------
        # エージェントのゴールまでの履歴を元に、行動方策を更新
        #------------------------------------------------------------
        new_policy = self._brain.decision_policy()

        #------------------------------------------------------------
        # エピソードの完了判定処理
        #------------------------------------------------------------
        # 前回の行動方針との差分が十分小さくなれば学習を終了する。
        delta_policy = np.sum( np.abs( new_policy - policy ) )
        print( "前回の行動方針との差分：", delta_policy )

        if( delta_policy < stop_epsilon ):
            done = True


        return