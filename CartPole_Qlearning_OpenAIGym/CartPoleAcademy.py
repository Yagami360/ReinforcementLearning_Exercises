# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#from JSAnimation.IPython_display import display_animation
#from IPython.display import display
import os.path

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

        _frames : list<>
            動画のフレーム（１つの要素が１画像のフレーム）
        
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, env, max_episode = 1, max_time_step = 100, save_step = 100 ):
        super().__init__( max_episode, max_time_step, save_step )
        self._env = env
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
        学習環境を実行する
        """
        # エピソードを試行
        for episode in range( 0, self._max_episode ):
            # 学習環境を RESET
            self.academy_reset()

            # 時間ステップを 1ステップづつ進める
            for time_step in range( 0 ,self._max_time_step ):
                dones = []

                if( episode % self._save_step == 0 ):
                    # 学習環境の動画のフレームを追加
                    self.add_frame( episode, time_step )
                if( episode == self._max_episode - 1 ):
                    # 学習環境の動画のフレームを追加
                    self.add_frame( episode, time_step )

                for agent in self._agents:
                    done = agent.agent_step( episode, time_step )
                    dones.append( done )

                # 全エージェントが完了した場合
                if( all(dones) == True ):
                    break

            # Academy と全 Agents のエピソードを完了
            self._done = True
            for agent in self._agents:
                agent.agent_on_done( episode, time_step )

            # 動画を保存
            if( episode % self._save_step == 0 ):
                self.save_frames( "RL_ENV_{}_Episode{}.gif".format(self._env.spec.id, episode) )
                self._frames = []

            if( episode == self._max_episode - 1 ):
                self.save_frames( "RL_ENV_{}_Episode{}.gif".format(self._env.spec.id, episode) )
                self._frames = []

        return


    def add_frame( self, episode, times_step ):
        """
        強化学習環境の１フレームを追加する
        """
        frame = self._env.render( mode='rgb_array' )
        self._frames.append( frame )

        return

    def save_frames( self, file_name = "RL_ENV_CartPole-v0.mp4" ):
        """
        外部ファイルに動画を保存する。
        """
        plt.clf()
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
        ftitle, fext = os.path.splitext(file_name)
        if( fext == ".gif" ):
            anim.save( file_name, writer = 'imagemagick' )
        else:
            anim.save( file_name )

        plt.close()

        return
