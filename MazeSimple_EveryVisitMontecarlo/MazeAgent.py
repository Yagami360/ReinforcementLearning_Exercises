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
        self._s_a_r_historys = [ [ self._state, self._action, 0.0 ] ]
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
        print( "_total_reward : \n", self._total_reward )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_state : \n", self._state )
        print( "_action : \n", self._action )
        print( "_s_a_r_historys : \n", self._s_a_r_historys )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def get_q_function_historys( self ):
        return self._q_function_historys

    def get_v_function_historys( self ):
        return self._v_function_historys


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._done = False
        self._total_reward = 0.0
        
        # s0 : エージェントの状態を再初期化 s0 して、開始位置に設定する。
        # a0 はまだ分からないので、np.nan
        # _s_a_historys = [s0, np.nan]
        self._state = self._s_a_r_historys[0][0]
        self._action = self._s_a_r_historys[0][1]
        self._s_a_r_historys = [ [ self._state, self._action ] ]
        self._s_a_r_historys = [ [ self._state, self._action, self._total_reward ] ]

        # a0 : 初期行動a0 を設定する。（Brain のロジックに従ったエージェント次の行動）
        self._action = next_action = self._brain.action( state = self._state )
        self._s_a_historys[-1][1] = self._action  # -1 で末端に追加
        self._s_a_r_historys[-1][1] = self._action  # -1 で末端に追加
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
            #next_action = np.nan
            next_action = 0
        else:
            next_action = self._brain.action( state = next_state )

        # Brain の更新処理
        #self._brain.update_q_function()

        # a' → r'' : 次行動 a' に対する報酬 r'' の指定
        reward = 0.0
        if( next_state == 8 ):
            reward = 1.0
            #self.add_reward( reward, time_step )  # ゴール地点なら、報酬１
        else:
            # ゴール地点でないなら、負の報酬（）
            reward = -0.01
            #self.add_reward( reward, time_step )
        
        # (s,a,r) の履歴を追加
        self._s_a_historys.append( [next_state, next_action] )
        self._s_a_r_historys.append( [next_state, next_action, reward] )

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


    def agent_on_done( self, episode, time_step ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数
        """
        #------------------------------------------------------------
        # エピソード完了後の処理
        #------------------------------------------------------------
        print( "エピソード = {0} / 最終時間ステップ数 = {1}".format( episode, time_step )  )
        print( "迷路を解くのにかかったステップ数：" + str( len(self._s_a_r_historys) ) )

        for (t,s_a_r) in enumerate( self._s_a_r_historys ):
            reward = s_a_r[2]
            self.add_reward( reward, time_step = t )

        # このメソッドが呼び出される度に、ε の値を徐々に小さくする。
        self._brain.decay_epsilon()

        # 利得の履歴に追加
        self._reward_historys.append( self._total_reward )

        # 逐次訪問MC法による方策評価
        self._brain.update_q_function( self._s_a_r_historys, self._total_reward )

        #---------------------------------------------
        # Q関数とV関数
        #---------------------------------------------
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

