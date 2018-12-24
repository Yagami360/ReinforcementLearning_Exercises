# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from IPython.display import HTML

# 自作モジュール
from Academy import Academy
from MazeAcademy import MazeAcademy

from Agent import Agent
from MazeAgent import MazeAgent

from Brain import Brain
from MazePolicyGradientBrain import MazePolicyGradientBrain



def main():
    """
	強化学習の学習環境用の迷路探索問題
    ・エージェントの行動方策の学習ロジックは、方策勾配法
    """
    print("Start main()")

    #-----------------------------------
    # 学習環境、エージェント生成フェイズ
    #-----------------------------------
    # Academy の生成
    academy = MazeAcademy( max_episode = 5000 )

    # Brain の生成
    brain = MazePolicyGradientBrain( learning_rate = 0.1 )

	# Agent の生成
    agent = MazeAgent()

    # Agent の Brain を設定（相互参照）
    agent.set_brain( brain )
    brain.set_agent( agent )

    # 学習環境に作成したエージェントを追加
    academy.add_agent( agent )
    
    agent.print( "after init()" )
    brain.print( "after init()" )

    #-----------------------------------
    # シミュレーションフェイズ
    #-----------------------------------
    academy.academy_step()
    agent.print( "after simulation" )
    brain.print( "after simulation" )

    #-----------------------------------
    # 描写処理
    #-----------------------------------
    # 初期位置での迷路の様子

    # 図を描く大きさと、図の変数名を宣言
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # 赤い壁を描く
    ax.plot([1, 1], [0, 1], color='red', linewidth=2)
    ax.plot([1, 2], [2, 2], color='red', linewidth=2)
    ax.plot([2, 2], [2, 1], color='red', linewidth=2)
    ax.plot([2, 3], [1, 1], color='red', linewidth=2)

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
    ax.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
        
    # 現在地S0に緑丸を描画する
    line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

    #
    #plt.savefig( "MazeGame_PolicyGradient.png", dpi = 300, bbox_inches = "tight" )
        
    #----------------------------------------
    # エージェントの移動の様子を可視化
    #----------------------------------------
    def init():
        '''背景画像の初期化'''
        line.set_data([], [])
        return (line,)

    def animate(i):
        '''フレームごとの描画内容'''
        state_history = agent.collect_observations()[1]
        state = state_history[i]  # 現在の場所を描く
        x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
        y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
        line.set_data(x, y)
        return (line,)

    #　初期化関数とフレームごとの描画関数を用いて動画を作成する
    anim = animation.FuncAnimation(
        fig, animate, 
        init_func=init, 
        frames=len( agent.collect_observations()[1] ), 
        interval=200, repeat=False
    )

    HTML( anim.to_jshtml() )
    anim.save( "MazeGame_PolicyGradient.gif", writer = 'imagemagick' )

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()

