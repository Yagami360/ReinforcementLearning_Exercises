# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt

# OpenAI Gym
import gym

# 動画の描写関数用
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
#from IPython.display import display

# 自作クラス
from Academy import Academy
from Agent import Agent


class CartPoleAcademy( Academy ):
    """
    エージェントの強化学習環境
    ・強化学習モデルにおける環境 Enviroment に対応
    ・学習や推論を行うための設定を行う。
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _max_episode : int
                       エピソードの最大回数
                       最大回数数に到達すると、Academy と全 Agent のエピソードを完了する。
        _max_time_step : int
                        時間ステップの最大回数

        _env : OpenAI Gym の ENV

        _agents : list<AgentBase>

        _done : bool
            エピソードが完了したかのフラグ

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, max_episode = 1, max_time_step = 100 ):
        self._max_episode = max_episode
        self._max_time_step = max_time_step
        self._env = gym.make( "CartPole-v0" )
        self._agents = []
        self._done = False
        self._frames = []
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


    def academy_reset( self ):
        """
        学習環境をリセットする。
        ・エピソードの開始時にコールされる
        """
        if( self._agents != None ):
            for agent in self._agents:
                agent.agent_reset()        

        self._done = False
        observations = self._env.reset()
        #print( "observations :", observations )

        for agent in self._agents:
            agent.set_observations( observations )

        return


    def academy_run( self ):
        """
        """
        # エピソードを試行
        for episode in range( 0, self._max_episode ):
            # 学習環境を RESET
            self.academy_reset()

            # 時間ステップを 1ステップづつ進める
            for time_step in range( 0 ,self._max_time_step ):
                # 学習環境の動画のフレームを追加
                #self._frames.append( self._env.render( module="rgb_array") )

                for agent in self._agents:
                    agent.agent_step( episode, time_step )

            # Academy と全 Agents のエピソードを完了
            self._done = True
            for agent in self._agents:
                agent.agent_on_done()

        return


    def display_frames_as_gif( self ):
        """
        Displays a list of frames as a gif, with controls
        """
        plt.figure(
            figsize=( self._frames[0].shape[1]/72.0, self._frames[0].shape[0]/72.0 ),
            dpi=72
        )
        patch = plt.imshow( self._frames[0] )
        plt.axis('off')

        def animate(i):
            patch.set_data( self._frames[i] )

        anim = animation.FuncAnimation(
                   plt.gcf(), 
                   animate, 
                   frames = len( self._frames ),
                   interval=50
        )

        anim.save( 'RL_ENV_{}.mp4'.format( "CartPole-v0" ) )  # 追記：動画の保存です
        #display( display_animation(anim, default_mode='loop') ) 
        return
