# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# OpenAI Gym
import gym

# 自作クラス
from Agent import Agent


class FrozenLakeAgent( Agent ):
    """
    OpenAIGym の CartPole のエージェント

    [protected] 変数名の前にアンダースコア _ を付ける
        _env : OpenAI Gym の ENV
        _q_function_historys : list< [int, float, float] > / shape = [n_episode, n_state, n_action]
            各エピソード完了後の Q 関数の値の履歴
        _v_function_historys : list< [int, float] > / shape = [n_episode, n_state]
            各エピソード完了後の V 関数の値の履歴
    """
    def __init__( 
        self,
        env,
        brain = None, 
        gamma = 0.9
    ):
        super().__init__( brain, gamma, 0 )
        self._env = env        
        self._observations = self._env.reset()
        self._q_function_historys = []
        self._v_function_historys = []
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "FrozenLakeAgent" )
        print( self )
        print( str )
        print( "_env :", self._env )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_total_reward : \n", self._total_reward )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_s_a_historys : \n", self._s_a_historys )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def get_q_function_historys( self ):
        return self._q_function_historys

    def get_v_function_historys( self ):
        return self._v_function_historys

    def get_num_states( self ):
        """
        エージェントの状態数を取得する
        """
        num_states = self._env.observation_space.shape.n
        return num_states

    def get_num_actions( self ):
        """
        エージェントの状態数を取得する
        """
        num_actions = self._env.action_space.n
        return num_actions

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._observations = self._env.reset()
        self._total_reward = 0.0
        self._done = False
        self._s_a_historys = [ [ self._state, self._action ] ]
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

        #-------------------------------------------------------------------
        # 離散化した現在の状態 s_t を求める
        #-------------------------------------------------------------------
        state = self._observations

        #-------------------------------------------------------------------
        # 離散化した現在の状態 s_t を元に、行動 a_t を求める
        #-------------------------------------------------------------------
        #self._brain.decay_epsilon()
        action = self._brain.action( state )
    
        #-------------------------------------------------------------------
        # 行動の実行により、次の時間での状態 s_{t+1} と報酬 r_{t+1} を求める。
        #-------------------------------------------------------------------
        observations_next, reward, env_done, info = self._env.step( action )
        next_state = observations_next
        #print( "env_done :", env_done )
        #print( "info :", info )

        # 次の状態 s' と次の行動 a' を履歴に追加
        self._s_a_historys.append( [next_state, action] )

        #----------------------------------------
        # 報酬の設定
        #----------------------------------------
        reward = 0.0
        if( env_done == True ):
            # ゴールしたとき
            if( next_state == 15 ):
                reward = 10.0
            else:
                reward = -10.0

        self.add_reward( reward, time_step )

        #----------------------------------------
        # 価値関数の更新
        #----------------------------------------
        self._brain.update_q_function( state, action, next_state, reward )

        #----------------------------------------
        # 状態の更新
        #----------------------------------------
        self._observations = observations_next
        self._state = next_state
        self._action = action

        #----------------------------------------
        # 完了時の処理
        #----------------------------------------
        if( env_done == True ):
            self.done()
        
        return self._done

    def agent_on_done(self, episode, time_step ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数
        """
        print( "エピソード = {0} / 最終時間ステップ数 = {1}".format( episode, time_step )  )
        print( "迷路を解くのにかかったステップ数：" + str( len(self._s_a_historys) ) )

        # 利得の履歴に追加
        self._reward_historys.append( self._total_reward )

        #--------------------------------------------------------------
        # Q関数、V関数の算出と、履歴の追加
        #--------------------------------------------------------------
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
