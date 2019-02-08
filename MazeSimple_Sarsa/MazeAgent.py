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
    ・迷宮探索の t→t+1 の１ステップ毎に、価値関数 Q を更新する。
    ・価値反復法（Sarsa, Q学習など）での使用を想定している。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _q_function_historys : list< [int, float, float] > / shape = [n_episode, n_state, n_action]
            各エピソード完了後の Q 関数の値の履歴
        _v_function_historys : list< [int, float] > / shape = [n_episode, n_state]
            各エピソード完了後の V 関数の値の履歴

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        brain = None, 
        gamma = 0.9, 
        state0 = 0
    ):
        super().__init__( brain, gamma, state0 )
        self._q_function_historys = []
        self._v_function_historys = []
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
        #print( "_q_function_historys : \n", self._q_function_historys )
        #print( "_v_function_historys : \n", self._v_function_historys )
        print( "----------------------------------" )
        return

    def get_q_function_historys( self ):
        return self._q_function_historys

    def get_v_function_historys( self ):
        return self._v_function_historys

    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        self._observations = []
        self.add_vector_obs( self._state )
        self.add_vector_obs( self._s_a_historys )
        self.add_vector_obs( self._q_function_historys )
        self.add_vector_obs( self._v_function_historys )
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
        
        # r0 : 初期の行動に対しての報酬 r1 の設定
        if( self._state == 8 ):
            self.add_reword( 1.0 )  # ゴール地点なら、報酬１
        else:
            self.add_reword( 0.0 )  # ゴール地点でないなら、報酬なし


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
            return self.done

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
            self.add_reword( 1.0 )  # ゴール地点なら、報酬１
        else:
            self.add_reword( 0.0 )  # ゴール地点なら、報酬なし


        # s,a,s',r',a' → Q : Q 関数を更新
        self._brain.update_q_function(
            state = self._state,
            action = self._action,
            next_state = next_state,
            next_action = next_action,
            reword = self._reword
        )

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

        # このメソッドが呼び出される度に、ε の値を徐々に小さくする。
        #self._brain.decay_learning_rate()
        self._brain.decay_epsilon()

        q_function = self._brain.get_q_function()

        # エピソード開始の価値関数との差分
        if( episode == 0 ):  
            # 初回エピソードの場合は、履歴に前回値がないので、初期値を push
            # deep copy したものを append
            copy_q_function = self._brain.get_q_function().copy()
            self._q_function_historys.append( copy_q_function )
            copy_v_function = np.nanmax( copy_q_function, axis = 1 )
            self._v_function_historys.append( copy_v_function )

        #delta_q_function = np.sum( np.abs( q_function - self._q_function_historys[-1] ) )
        #print( "エピソードの Q 関数との差分：", delta_q_function )
        
        # 状態価値関数 V の算出
        new_v_function = np.nanmax( q_function, axis = 1 )
        v_function = np.nanmax( self._q_function_historys[-1], axis = 1 )
        delta_v_function = np.sum( np.abs( new_v_function - v_function ) )
        print( "V 関数の大きさ：", np.abs( new_v_function ) )
        print( "前回のエピソードの V 関数との差分：", delta_v_function )

        # エピソード完了後の価値関数の値を保管
        # deep copy したものを append
        self._q_function_historys.append( q_function.copy() )
        self._v_function_historys.append( new_v_function )

        return

