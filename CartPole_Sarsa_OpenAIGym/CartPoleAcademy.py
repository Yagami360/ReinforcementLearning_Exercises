# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt

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
        _env : OpenAIGym の ENV

        _max_episode : int
                       エピソードの最大回数
                       最大回数数に到達すると、Academy と全 Agent のエピソードを完了する。
        _max_time_step : int
                        時間ステップの最大回数

        _agents : list<AgentBase>

        _done : bool
            エピソードが完了したかのフラグ

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, env, max_episode = 1, max_time_step = 100 ):
        self._env = env
        self._max_episode = max_episode
        self._max_time_step = max_time_step
        self._agents = []
        self._done = False
        self._frames = []
        return


    def academy_reset( self ):
        """
        学習環境をリセットする。
        ・エピソードの開始時にコールされる
        """
        if( self._agents != None ):
            for agent in self._agents:
                agent.agent_reset()        

        self._done = False
        self._env.reset()
        return

    def academy_run( self ):
        """
        """
        # エピソードを試行
        for episode in range( 0, self._max_episode ):
            print( "現在のエピソード数：", episode )

            # 学習環境を RESET
            self.academy_reset()

            # 時間ステップを 1ステップづつ進める
            for time_step in range( 0 ,self._max_time_step ):
                # 学習環境の動画のフレームを追加
                self.add_frame()

                for agent in self._agents:
                    done = agent.agent_step( episode, time_step )

                if( done == True ):
                    break

            # Academy と全 Agents のエピソードを完了
            self._done = True
            for agent in self._agents:
                agent.agent_on_done()

        return


    def add_frame( self ):
        """
        強化学習環境の１フレームを追加する
        """
        frame = self._env.render( mode = "rgb_array" )
        self._frames.append( frame )
        return

    def display_frames( self, file_name = "RL_ENV_CartPole-v0.mp4" ):
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

        # 動画の保存
        anim.save( file_name )
        #display( display_animation(anim, default_mode='loop') ) 
        return
