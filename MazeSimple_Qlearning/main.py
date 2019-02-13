# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from IPython.display import HTML

# 自作モジュール
from Academy import Academy
from MazeAcademy import MazeAcademy
from Brain import Brain
from MazeQlearningBrain import MazeQlearningBrain
from Agent import Agent
from MazeAgent import MazeAgent

# 設定可能な定数
NUM_EPISODE = 100           # エピソード試行回数
NUM_TIME_STEP = 500         # １エピソードの時間ステップの最大数
AGANT_NUM_STATES = 8        # 状態の要素数（s0~s7）※ 終端状態 s8 は除いた数
AGANT_NUM_ACTIONS = 4       # 行動の要素数（↑↓→←）
AGENT_INIT_STATE = 0        # 初期状態の位置 0 ~ 8
BRAIN_LEARNING_RATE = 0.1   # 学習率
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.9          # 割引率


def main():
    """
	強化学習の学習環境用の迷路探索問題
    ・エージェントの行動方策の学習ロジックは、Q学習
    """
    print("Start main()")

    #===================================
    # 学習環境、エージェント生成フェイズ
    #===================================
    #-----------------------------------
    # Academy の生成
    #-----------------------------------
    academy = MazeAcademy( max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP )

    #-----------------------------------
    # Brain の生成
    #-----------------------------------
    # 行動方策のためのパラメーターを表形式（行：状態 s、列：行動 a）で定義
    # ※行動方策を表形式で実装するために、これに対応するパラメーターも表形式で実装する。
    # 進行方向に壁があって進めない様子を表現するために、壁で進めない方向には `np.nan` で初期化する。
    # 尚、状態 s8 は、ゴール状態で行動方策がないため、これに対応するパラメーターも定義しないようにする。
    brain_parameters = np.array(
        [   # a0="Up", a1="Right", a3="Down", a4="Left"
            [ np.nan, 1,        1,         np.nan ], # s0
            [ np.nan, 1,        np.nan,    1 ],      # s1
            [ np.nan, np.nan,   1,         1 ],      # s2
            [ 1,      1,        1,         np.nan ], # s3
            [ np.nan, np.nan,   1,         1 ],      # s4
            [ 1,      np.nan,   np.nan,    np.nan ], # s5
            [ 1,      np.nan,   np.nan,    np.nan ], # s6
            [ 1,      1,        np.nan,    np.nan ], # s7
            #[ np.nan, np.nan,   np.nan,    1 ],      # s8
        ]
    )

    brain1 = MazeQlearningBrain(
        n_states = AGANT_NUM_STATES,
        n_actions = AGANT_NUM_ACTIONS,
        brain_parameters = brain_parameters,
        epsilon = BRAIN_GREEDY_EPSILON,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE
    )
    
    #-----------------------------------
	# Agent の生成
    #-----------------------------------
    agent1 = MazeAgent(
        brain = brain1,
        gamma = BRAIN_GAMMDA,
        state0 = AGENT_INIT_STATE
    )

    # Agent の Brain を設定
    agent1.set_brain( brain1 )

    # 学習環境に作成したエージェントを追加
    academy.add_agent( agent1 )
    
    agent1.print( "after init()" )
    brain1.print( "after init()" )

    #===================================
    # エピソードの実行
    #===================================
    academy.academy_run()
    agent1.print( "after simulation" )
    brain1.print( "after simulation" )

    #===================================
    # 学習結果の描写処理
    #===================================
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
    #plt.savefig( "MazeGame_Sarsa.png", dpi = 300, bbox_inches = "tight" )
        
    #----------------------------------------
    # エージェントの移動の様子を可視化
    #----------------------------------------
    def init():
        '''背景画像の初期化'''
        line.set_data([], [])
        return (line,)

    def animate(i):
        '''フレームごとの描画内容'''
        s_a_historys = agent1.get_s_a_historys()
        state = s_a_historys[i][0]  # 現在の場所を描く
        x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
        y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
        line.set_data(x, y)
        return (line,)

    #　初期化関数とフレームごとの描画関数を用いて動画を作成する
    s_a_historys = agent1.get_s_a_historys()
    anim = animation.FuncAnimation(
        fig, animate, 
        init_func = init, 
        frames = len( s_a_historys ), 
        interval = 200, repeat = False
    )

    HTML( anim.to_jshtml() )
    anim.save( "MazeSimple_Qlearning_episode{}.gif".format(NUM_EPISODE), writer = 'imagemagick' )


    #---------------------------------------------
    # 状態 s0 ~ s7 での状態価値関数の値を plot
    # ※ 状態 s8 は、ゴール状態で行動方策がないため、これに対応する状態価値関数も定義されない。
    #---------------------------------------------
    # 各エピソードでの状態価値関数
    v_function_historys1 = agent1.get_v_function_historys()
    
    # list<ndarray> / shape=[n_episode,n_state] 
    # → list[ndarray] / shape = [n_episode,]
    v_function_historys1_s0 = []
    v_function_historys1_s1 = []
    v_function_historys1_s2 = []
    v_function_historys1_s3 = []
    v_function_historys1_s4 = []
    v_function_historys1_s5 = []
    v_function_historys1_s6 = []
    v_function_historys1_s7 = []
    v_function_historys1_s8 = []

    for v_function in v_function_historys1 :
        v_function_historys1_s0.append( v_function[0] )
        v_function_historys1_s1.append( v_function[1] )
        v_function_historys1_s2.append( v_function[2] )
        v_function_historys1_s3.append( v_function[3] )
        v_function_historys1_s4.append( v_function[4] )
        v_function_historys1_s5.append( v_function[5] )
        v_function_historys1_s6.append( v_function[6] )
        v_function_historys1_s7.append( v_function[7] )
        v_function_historys1_s8.append( 0 )

    # Q 学習での状態価値関数
    plt.clf()

    # S0
    plt.subplot( 3, 3, 1 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s0,
        label = 'S0 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S0" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    # S1
    plt.subplot( 3, 3, 2 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s1,
        label = 'S1 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S1" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    # S2
    plt.subplot( 3, 3, 3 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s2,
        label = 'S2 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S2" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    # S3
    plt.subplot( 3, 3, 4 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s3,
        label = 'S3 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S3" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    # S4
    plt.subplot( 3, 3, 5 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s4,
        label = 'S4 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S4" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()
    
    # S5
    plt.subplot( 3, 3, 6 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s5,
        label = 'S5 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S5" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    # S6
    plt.subplot( 3, 3, 7 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s6,
        label = 'S6 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S6" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    # S7
    plt.subplot( 3, 3, 8 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s7,
        label = 'S7 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S7" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()
    
    # S8
    plt.subplot( 3, 3, 9 )
    plt.plot(
        range(0,NUM_EPISODE+1), v_function_historys1_s8,
        label = 'S8 / Q-learning',
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "V function / S8" )
    plt.xlim( 0, NUM_EPISODE+1 )
    plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()

    plt.savefig( "MazaSimple_Q-learning_1-1_episode{}.png".format(NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
    plt.show()


    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


