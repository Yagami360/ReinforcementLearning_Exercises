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
from FrozenLakeRamdomBrain import FrozenLakeRamdomBrain
from Agent import Agent
from FrozenLakeAgent import FrozenLakeAgent


#--------------------------------
# 設定可能な定数
#--------------------------------
RL_ENV = "FrozenLakeNotSlippery-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 500           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
BRAIN_GAMMDA = 0.99         # 割引率

from gym.envs.registration import register

register(
    id = RL_ENV,   # 独自の Env の id
    entry_point = "gym.envs.toy_text:FrozenLakeEnv",
    kwargs = {"is_slippery": False}
)


def main():
    """
	強化学習の学習環境用のFrozenLake
    ・エージェントの行動方策の学習ロジックは、ランダム
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
    brain = FrozenLakeRamdomBrain(
        n_states = env.observation_space.n,
        n_actions = env.action_space.n
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

    plt.savefig( "{}_Reward_episode{}.png".format( RL_ENV, NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
    plt.show()

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
