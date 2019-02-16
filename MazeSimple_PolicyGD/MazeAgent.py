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


class MazeAgent( Agent ):
    """
    迷路探索用エージェント。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        brain = None, 
        gamma = 0.9, 
        state0 = 0
    ):
        super().__init__( brain, gamma, state0 )
        return


    def print( self, str ):
        print( "----------------------------------" )
        print( "MazeAgent" )
        print( self )
        print( str )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_reword : \n", self._reword )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_state : \n", self._state )
        print( "_action : \n", self._action )
        print( "_s_a_historys : \n", self._s_a_historys )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        self._observations = []
        self.add_vector_obs( self._state )
        self.add_vector_obs( self._s_a_historys )
        return self._observations


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._done = False
        self._reword = 0.0
        
        # s0 : エージェントの状態を再初期化 s0 して、開始位置に設定する。
        # a0 はまだ分からないので、np.nan
        # _s_a_historys = [s0, np.nan]
        self._state = self._s_a_historys[0][0]
        self._action = self._s_a_historys[0][1]
        self._s_a_historys = [ [ self._state, self._action ] ]

        # a0 : 初期行動a0 を設定する。（Brain のロジックに従ったエージェント次の行動）
        self._action = next_action = self._brain.action( state = self._state )
        self._s_a_historys[-1][1] = self._action  # -1 で末端に追加
        return

    def agent_step( self, episode, time_step ):
        """
        エージェント [Agent] の次の状態を決定する。
        ・Academy から各時間ステップ度にコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
            time_step : 現在の時間ステップ

        [Returns]
            done : bool
                   エピソードの完了フラグ
        """
        # 既にエピソードが完了状態なら、そのまま return して、全エージェントの完了を待つ
        if( self._done == True):
            return self._done

        #print( "現在のエピソード数：", episode )
        #print( "現在の時間ステップ数：", time_step )

        #----------------------------------------------------------------------
        # １時間ステップでの迷宮探索
        # バックアップ線図 : s_t → a_t → r_(t+1) → s_(t+1) → a_(t+1) → r_(t+2) → Q → ...
        #----------------------------------------------------------------------
        # r → a → s' : 行動 a に従った次状態 s' の決定
        if self._action == 0:    # Up
            next_state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
        elif self._action == 1:  # Right
            next_state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
        elif self._action == 2:  # Down
            next_state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
        elif self._action == 3:  # Left
            next_state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる
        else:   # np.nan など
            next_state = self._state

        # s' → a' : 次状態 s' での次行動 a'
        if( next_state == 8 ):
            next_action = np.nan
        else:
            next_action = self._brain.action( state = next_state )

        # 次の状態 s' と次の行動 a' を履歴に追加
        self._s_a_historys.append( [next_state, next_action] )
        
        # a' → r'' : 次行動 a' に対する報酬 r'' の指定
        if( next_state == 8 ):
            # ゴール地点なら、報酬１
            self.add_reword( 1.0, time_step )
        else:
            # ゴール地点なら、小さな負の報酬
            self.add_reword( -0.01, time_step )
        
        # ゴールの指定
        if( next_state == 8 ):
            self._state = next_state
            self._action = next_action
            self._done = True              

        else:
            # 状態と行動を更新：s←s', a←a'
            self._state = next_state
            self._action = next_action
            self._done = False              

        return self._done


    def agent_on_done( self, episode ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数
        """
        #------------------------------------------------------------
        # １エピソード完了後の処理
        #------------------------------------------------------------
        print( "迷路を解くのにかかったステップ数：" + str( len(self._s_a_historys) ) )

        # 利得の履歴に追加
        self._reward_historys.append( self._reword )

        #------------------------------------------------------------
        # エージェントのゴールまでの履歴を元に、行動方策を更新
        #------------------------------------------------------------
        policy = self._brain.get_policy()
        self._brain.update( self._s_a_historys )
        new_policy = self._brain.get_policy()

        #------------------------------------------------------------
        # 学習の完了判定処理
        #------------------------------------------------------------
        stop_epsilon = 0.001    # 学習完了のための行動方策の差分値

        # 前回の行動方針との差分が十分小さくなれば学習を終了する。
        delta_policy = np.sum( np.abs( new_policy - policy ) )
        print( "前回の行動方針との差分：", delta_policy )

        if( delta_policy < stop_epsilon ):
            pass    # 予約コード（何かしらの学習完了処理）

        return

