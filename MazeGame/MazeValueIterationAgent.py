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


class MazeValueIterationAgent( Agent ):
    """
    迷路探索用エージェント。
    ・迷宮探索の t→t+1 の１ステップ毎に、価値関数 Q を更新する。
    ・価値反復法（Sarsa, Q学習など）での使用を想定している。

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
        print( "MazeValueIterationAgent" )
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

        #----------------------------------------------------------------------
        # 行動方策に基づき、エージェントを迷路のゴールまで移動させる。
        # ※ 迷路のスタートからゴールまでを、１エピソードとする。
        # バックアップ線図 : s0 → a0 → ... s → a → r' → s' → a' → r'' → Q → ...
        #----------------------------------------------------------------------
        # s0 : エージェントの状態を再初期化 s0 して、開始位置に設定する。
        # a0 はまだ分からないので、np.nan
        # _s_a_historys = [s0, np.nan]
        self.agent_reset()
        
        # a0 : 初期行動a0 を設定する。（Brain のロジックに従ったエージェント次の行動）
        self._action = next_action = self._brain.action( state = self._state )
        self._s_a_historys[-1][1] = self._action  # -1 で末端に追加
        
        # r0 : 初期の行動に対しての報酬 r1 の設定
        if( self._state == 8 ):
            self.add_reword( 1.0 )  # ゴール地点なら、報酬１
        else:
            self.add_reword( 0.0 )  # ゴール地点なら、報酬なし

        # Goal にたどり着くまでループ
        # バックアップ線図 : ... → s → a → r' → s' → a' → r'' → Q → while ループ...
        while(1):
            # r → a → s' : 行動 a に従った次状態 s' の決定
            if self._action == 0:    # Up
                next_state = self._state - 3  # 上に移動するときは状態の数字が3小さくなる
            elif self._action == 1:  # Right
                next_state = self._state + 1  # 右に移動するときは状態の数字が1大きくなる
            elif self._action == 2:  # Down
                next_state = self._state + 3  # 下に移動するときは状態の数字が3大きくなる
            elif self._action == 3:  # Left
                next_state = self._state - 1  # 左に移動するときは状態の数字が1小さくなる

            # s' → a' : 次状態 s' での次行動 a'
            if( next_state == 8 ):
                next_action = np.nan
            else:
                next_action = self._brain.action( state = next_state )

            # 次の状態 s' と次の行動 a' を履歴に追加
            self._s_a_historys.append( [next_state, next_action] )

            # a' → r'' : 次行動 a' に対する報酬 r'' の指定
            if( next_state == 8 ):
                self.add_reword( 1.0 )  # ゴール地点なら、報酬１
            else:
                self.add_reword( 0.0 )  # ゴール地点なら、報酬なし


            # s,a,s',r',a' → Q : Q 関数を更新
            q_function = self._brain.update_q_function(
                state = self._state,
                action = self._action,
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
                self._action = next_action

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