# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
#from IPython.display import HTML

# OpenAI Gym
import gym

# 動画の描写関数用
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

# 自作モジュール
from Academy import Academy
from CartPoleAcademy import CartPoleAcademy
from Brain import Brain
from CartPoleQlearningBrain import CartPoleQlearningBrain
from Agent import Agent
from CartPoleAgent import CartPoleAgent


#--------------------------------
# 設定可能な定数
#--------------------------------
RL_ENV = "CartPole-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 500           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
NUM_DIZITIZED = 6           # 各状態の離散値への分割数
BRAIN_LEARNING_RATE = 0.5   # 学習率
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99         # 割引率


def main():
    """
	強化学習の学習環境用の倒立振子課題 CartPole
    ・エージェントの行動方策の学習ロジックは、Qlearning
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

    #-----------------------------------
    # Academy の生成
    #-----------------------------------
    academy = CartPoleAcademy( env = env, max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP, save_step = 100 )

    #-----------------------------------
    # Brain の生成
    #-----------------------------------
    brain = CartPoleQlearningBrain(
        n_states = env.observation_space.shape[0],
        n_actions = env.action_space.n,
        epsilon = BRAIN_GREEDY_EPSILON,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE,
        n_dizitzed = NUM_DIZITIZED
    )

    #-----------------------------------
	# Agent の生成
    #-----------------------------------
    agent = CartPoleAgent(
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
    #academy.save_frames( file_name = "RL_ENV_CartPole-v0.mp4" )

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

    plt.savefig( "{}_Qlearning_Reward_episode{}.png".format( RL_ENV, NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
    plt.show()

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


