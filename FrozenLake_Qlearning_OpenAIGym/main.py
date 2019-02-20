# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation    # 動画の描写関数用

# OpenAI Gym
import gym
from gym import wrappers

# 自作モジュール
from Academy import Academy
from FrozenLakeAcademy import FrozenLakeAcademy
from Brain import Brain
from FrozenLakeQlearningBrain import FrozenLakeQlearningBrain
from Agent import Agent
from FrozenLakeAgent import FrozenLakeAgent


#--------------------------------
# 設定可能な定数
#--------------------------------
RL_ENV = "FrozenLakeNotSlippery-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 500           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.1   # 学習率
BRAIN_GREEDY_EPSILON = 0.01  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99          # 割引率

from gym.envs.registration import register

register(
    id = RL_ENV,   # 独自の Env の id
    entry_point = "gym.envs.toy_text:FrozenLakeEnv",
    kwargs = {"is_slippery": False}
)

def main():
    """
	強化学習の学習環境用のFrozenLake
    ・エージェントの行動方策の学習ロジックは、Q学習
    """
    print("Start main()")
    
    # バージョン確認
    print( "OpenAI Gym", gym.__version__ )

    np.random.seed(8)
    random.seed(8)

    #===================================
    # 学習環境、エージェント生成フェイズ
    #===================================
    # OpenAI-Gym の ENV を作成
    env = gym.make( RL_ENV )
    #env = wrappers.Monitor( env, directory = "/tmp/frozenlake-v0", force = True )
    env.seed(8)

    #-----------------------------------
    # Academy の生成
    #-----------------------------------
    academy = FrozenLakeAcademy( env = env, max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP, save_step = 1 )

    #-----------------------------------
    # Brain の生成
    #-----------------------------------
    brain = FrozenLakeQlearningBrain(
        n_states = env.observation_space.n,
        n_actions = env.action_space.n,
        epsilon = BRAIN_GREEDY_EPSILON,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE
    )

    #-----------------------------------
	# Agent の生成
    #-----------------------------------
    agent = FrozenLakeAgent(
        env = env,
        brain = brain,
        gamma = BRAIN_GAMMDA
    )

    # Agent の Brain を設定
    agent.set_brain( brain )

    # 学習環境に作成したエージェントを追加
    academy.add_agent( agent )
    
    agent.print( "after init()" )
    brain.print( "after init()" )

    #===================================
    # エピソードの実行
    #===================================
    academy.academy_run()
    agent.print( "after run" )
    brain.print( "after run" )

    #===================================
    # 学習結果の描写処理
    #===================================
    #---------------------------------------------
    # 利得の履歴の plot
    #---------------------------------------------
    reward_historys = agent.get_reward_historys()

    plt.clf()
    plt.plot(
        range(0,NUM_EPISODE+1), reward_historys,
        label = 'gamma = {}'.format(BRAIN_GAMMDA),
        linestyle = '-',
        linewidth = 0.5,
        color = 'black'
    )
    plt.title( "Reward History" )
    plt.xlim( 0, NUM_EPISODE+1 )
    #plt.ylim( [-0.1, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.legend( loc = "lower right" )
    plt.tight_layout()

    plt.savefig( "{}_Reward_episode{}.png".format( RL_ENV, NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
    plt.show()

    #---------------------------------------------
    # 行動価値関数を plot
    #---------------------------------------------
    Q_function_historys = agent.get_q_function_historys()
    Q_function = Q_function_historys[-1]

    def draw_q_function( q_func ):
        """
        Q関数をグリッド上に分割したヒータマップで描写する。
        |　|↑　|　|
        |←|平均|→|
        |　|↓|　|
        """
        import matplotlib.cm as cm  # color map

        n_row = env.unwrapped.nrow   # Maze の行数
        n_col = env.unwrapped.ncol   # Maze の列数
        n_qrow = n_row * 3
        n_qcol = n_col * 3
        q_draw_map = np.zeros( shape = (n_qrow,n_qcol) )

        for i in range( n_row ):
            for j in range( n_col ):
                k = i * n_row + j   # 状態の格子番号
                _i = 1 + ( n_row - 1 - i ) * 3
                _j = 1 + j * 3
                q_draw_map[_i][_j-1] = q_func[k][0]     # Left
                q_draw_map[_i-1][_j] = q_func[k][1]     # Down
                q_draw_map[_i][_j+1] = q_func[k][2]     # Right
                q_draw_map[_i+1][_j] = q_func[k][3]     # Up
                q_draw_map[_i][_j] = np.mean( q_func[k] )

        q_draw_map = np.nan_to_num(q_draw_map)
        #print( "q_draw_map :", q_draw_map )

        fig = plt.figure()
        ax = fig.add_subplot( 1,1,1 )
        plt.imshow(
            q_draw_map,
            cmap = cm.RdYlGn,
            interpolation = "bilinear",
            vmax = abs( q_draw_map ).max(),
            vmin = -abs( q_draw_map ).max()
        )

        plt.colorbar()
        ax.set_xlim( -0.5, n_qcol - 0.5 )
        ax.set_ylim( -0.5, n_qrow - 0.5 )
        ax.set_xticks( np.arange(-0.5, n_qcol, 3) )
        ax.set_yticks( np.arange(-0.5, n_qrow, 3) )
        #ax.set_xticklabels( range(n_col+1) )
        #ax.set_yticklabels( range(n_row+1) )
        ax.grid( which = "both" )
        
        # 軸を消す
        plt.tick_params(
            axis='both', which='both', bottom='off', top='off',
            labelbottom='off', right='off', left='off', labelleft='off'
        )

        plt.title( "Q function" )
        plt.savefig( "{}_Qlearning_Qfunction_episode{}.png".format(RL_ENV, NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
        plt.show()

    draw_q_function( Q_function )

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
