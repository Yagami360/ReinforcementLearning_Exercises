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

        print( "現在のエピソード数：", episode )
        policy = self._brain.get_policy()

        #------------------------------------------------------------
        # 行動方策に基づき、エージェントを迷路のゴールまで移動させる。
        # ※ 迷路のスタートからゴールまでを、１エピソードとする。
        #------------------------------------------------------------
        # エージェントの状態を再初期化して、開始位置に設定する。
        self.agent_reset()
        
        # 初期行動を設定する。（Brain のロジックに従ったエージェント次の行動）
        action = next_action = self._brain.next_action( state = self._state )

        # Goal にたどり着くまでループ
        while(1):
            # 行動を更新
            action = next_action

            # 現在の状態 s の行動 a を設定
            self._s_a_historys[-1][1] = action  # -1 で末端に追加

            # エージェントの移動
            if action == 0:    # Up
                next_state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
            elif action == 1:  # Right
                next_state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
            elif action == 2:  # Down
                next_state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
            elif action == 3:  # Left
                next_state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる

            # 次の状態 s'を追加
            # 次の状態での行動はまだ分からないので NaN 値を入れておく。
            self._s_a_historys.append( [next_state, np.nan] )

            # 報酬の指定
            if( next_state == 8 ):
                self.add_reword( 1.0 )  # ゴール地点なら、報酬１
                next_action = np.nan
            else:
                self.add_reword( 0.0 )  # ゴール地点なら、報酬なし
                next_action = self._brain.next_action( state = next_state )

            # Q 関数を更新
            q_function = self._brain.update_q_function(
                state = self._state,
                action = action,
                next_state = next_state,
                next_action = next_action,
                reword = self._reword
            )

            # ゴールの指定
            if( next_state == 8 ):
                break              
            else:
                # 状態と行動を更新：s←s', a←a'
                self._state = next_state

        #------------------------------------------------------------
        # １エピソード完了後の処理
        #------------------------------------------------------------
        print( "迷路を解くのにかかったステップ数：" + str( len(self._s_a_historys) ) )

        # このメソッドが呼び出される度に、ε の値を徐々に小さくする。
        #self._brain.decay_learning_rate()
        self._brain.decay_epsilon()

        # 前回のエピソードの Q 関数との差分
        """
        delta_q_function = np.sum( np.abs( new_policy - policy ) )
        print( "前回のエピソードの Q 関数との差分：", delta_q_function )
        """

        return