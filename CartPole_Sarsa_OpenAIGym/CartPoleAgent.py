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


class CartPoleAgent( Agent ):
    """
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
        self._q_function_historys = []
        self._v_function_historys = []
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleAgent" )
        print( self )
        print( str )
        print( "_env :", self._env )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_reword : \n", self._reword )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_s_a_historys : \n", self._s_a_historys )
        #print( "_q_function_historys : \n", self._q_function_historys )
        #print( "_v_function_historys : \n", self._v_function_historys )
        print( "----------------------------------" )
        return

    def get_num_states( self ):
        """
        エージェントの状態数を取得する
        """
        num_states = self._env.observation_space.shape[0]
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
        self._reword = 0.0
        self._done = False
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
        done = False

        print( "現在のエピソード数：", episode )
        print( "現在の時間ステップ数：", time_step )

        #-------------------------------------------------------------------
        # 離散化した現在の状態 s_t を求める
        #-------------------------------------------------------------------
        state = self._brain.digitize_state( self._observations )
        print( "state :", state )

        #-------------------------------------------------------------------
        # 離散化した現在の状態 s_t を元に、行動 a_t を求める
        #-------------------------------------------------------------------
        action = self._brain.action( state )
        self._brain.decay_epsilon()
    
        #-------------------------------------------------------------------
        # 行動の実行により、次の時間での状態 s_{t+1} と報酬 r_{t+1} を求める。
        #-------------------------------------------------------------------
        observations_next, reward, env_done, info = self._env.step( action )
        next_state = self._brain.digitize_state( self._observations )
        print( "env_done :", env_done )
        print( "info :", info )

        #----------------------------------------
        # 報酬の設定
        #----------------------------------------
        if( env_done == True ):
            # 時間ステップの最大回数に近づいたら
            if time_step < 195:
                # 途中でコケたら、報酬－１
                self.add_reword( -1 )
            else:
                # 立ったまま終了時は、報酬＋１
                self.add_reword( 1 )
        else:
            # 途中報酬は０
            self.set_reword( 0 )

        #----------------------------------------
        # 価値関数の更新
        #----------------------------------------
        self._brain.update_q_function( state, action, next_state, self._reword )

        #----------------------------------------
        # 状態の更新
        #----------------------------------------
        state_next = state

        #----------------------------------------
        # 完了時の処理
        #----------------------------------------
        #if( env_done == True ):
            #self.done()
        
        return done
