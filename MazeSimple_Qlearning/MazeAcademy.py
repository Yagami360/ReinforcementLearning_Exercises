# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/05] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os.path

# 自作クラス
from Academy import Academy
from Agent import Agent


class MazeAcademy( Academy ):
    """
    迷宮問題のエージェントの学習環境
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
    
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, max_episode = 1, max_time_step = 100 ):
        super().__init__( max_episode )
        #self._frames = []
        return

    def academy_run(self):
        # エピソードを試行
        for episode in range( 0, self._max_episode ):
            print( "現在のエピソード数：", episode )

            # 学習環境を RESET
            self.academy_reset()

            # 時間ステップを 1ステップづつ進める
            for time_step in range( 0 ,self._max_time_step ):
                dones = []
                if( episode % 5 == 0 ):
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
                agent.agent_on_done( episode )

            # 動画を保存
            #if( episode % 5 == 0 ):
                #self.save_frames( "RL_ENV_SimpleMaze_Qlearning_Episode{}.gif".format(episode) )
                #self._frames = []

        return

    def add_frame(self, episode, times_step):
        return

    def save_frames(self, file_name):
        #----------------------------------------
        # エージェントの移動の様子を可視化
        #----------------------------------------
        # 初期位置での迷路の様子

        # 図を描く大きさと、図の変数名を宣言
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()

        # 壁を描く
        ax.plot([1, 1], [0, 1], color='black', linewidth=2)
        ax.plot([1, 2], [2, 2], color='black', linewidth=2)
        ax.plot([2, 2], [2, 1], color='black', linewidth=2)
        ax.plot([2, 3], [1, 1], color='black', linewidth=2)

        # 状態を示す文字S0～S8を描く
        ax.text(0.5, 2.5, 'S0', size=14, ha='center')
        ax.text(1.5, 2.5, 'S1', size=14, ha='center')
        ax.text(2.5, 2.5, 'S2', size=14, ha='center')
        ax.text(0.5, 1.5, 'S3', size=14, ha='center')
        ax.text(1.5, 1.5, 'S4', size=14, ha='center')
        ax.text(2.5, 1.5, 'S5', size=14, ha='center')
        ax.text(0.5, 0.5, 'S6', size=14, ha='center')
        ax.text(1.5, 0.5, 'S7', size=14, ha='center')
        ax.text(2.5, 0.5, 'S8', size=14, ha='center')
        ax.text(0.5, 2.3, 'START', ha='center')
        ax.text(2.5, 0.3, 'GOAL', ha='center')

        # 描画範囲の設定と目盛りを消す設定
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.tick_params(
            axis='both', which='both', bottom='off', top='off',
            labelbottom='off', right='off', left='off', labelleft='off'
        )
        
        # ゴール地点に緑枠


        # 現在地S0にエージェントを表す緑丸を描画する
        agent_image, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

        s_a_historys = self._agents[0].get_s_a_historys()

        def init():
            '''背景画像の初期化'''
            agent_image.set_data([], [])
            return (agent_image,)

        def animate(i):
            '''フレームごとの描画内容'''
            state = s_a_historys[i][0]  # 現在の場所を描く
            x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
            y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
            agent_image.set_data(x, y)
            return (agent_image,)

        #　初期化関数とフレームごとの描画関数を用いて動画を作成する
        anim = animation.FuncAnimation(
            fig, animate, 
            init_func = init, 
            frames = len( s_a_historys ), 
            interval = 200, repeat = False
        )

        # 動画の保存
        ftitle, fext = os.path.splitext(file_name)
        if( fext == ".gif" ):
            anim.save( file_name, writer = 'imagemagick' )
        else:
            anim.save( file_name )

        

        return

