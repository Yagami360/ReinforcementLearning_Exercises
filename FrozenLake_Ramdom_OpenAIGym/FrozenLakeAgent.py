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
        #self._brain.decay_epsilon( episode )
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
        #self._brain.update_q_function( state, action, next_state, reward )

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

        return
