# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation

# OpenAI Gym
import gym

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch  import nn   # ネットワークの構成関連
import torchvision      # 画像処理関連

# 自作モジュール
from Academy import Academy
from CartPoleAcademy import CartPoleAcademy
from Brain import Brain
from CartPoleDQN2013Brain import CartPoleDQN2013Brain
from CartPoleDQN2015Brain import CartPoleDQN2015Brain
from Agent import Agent
from CartPoleAgent import CartPoleAgent
from ExperienceReplay import ExperienceReplay

#--------------------------------
# 設定可能な定数
#--------------------------------
RL_ENV = "CartPole-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 500               # エピソード試行回数
NUM_TIME_STEP = 200             # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.0001    # 学習率
BRAIN_BATCH_SIZE = 32           # ミニバッチサイズ
BRAIN_GREEDY_EPSILON = 0.5      # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99             # 利得の割引率
MEMORY_CAPACITY = 10000         # Experience Relay 用の学習用データセットのメモリの最大の長さ


def main():
    """
	強化学習の学習環境用の倒立振子課題 CartPole
    ・エージェントの行動方策の学習ロジックは、DQN (2015年Natureバージョン)と203年バージョンの比較
    """
    print("Start main()")
    
    # バージョン確認
    print( "OpenAI Gym", gym.__version__ )
    print( "PyTorch :", torch.__version__ )

    np.random.seed(8)
    random.seed(8)
    torch.manual_seed(8)

    #===================================
    # 学習環境、エージェント生成フェイズ
    #===================================
    # OpenAI-Gym の ENV を作成
    env1 = gym.make( RL_ENV )
    env2 = gym.make( RL_ENV )
    env1.seed(8)
    env2.seed(8)

    #-----------------------------------
    # Academy の生成
    #-----------------------------------
    academy1 = CartPoleAcademy( env = env1, max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP, save_step = 500 )
    academy2 = CartPoleAcademy( env = env2, max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP, save_step = 500 )

    #-----------------------------------
    # Brain の生成
    #-----------------------------------
    brain1 = CartPoleDQN2015Brain(
        n_states = env1.observation_space.shape[0],
        n_actions = env1.action_space.n,
        epsilon = BRAIN_GREEDY_EPSILON,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE,
        batch_size = BRAIN_BATCH_SIZE,
        memory_capacity = MEMORY_CAPACITY
    )
    
    brain2 = CartPoleDQN2013Brain(
        n_states = env2.observation_space.shape[0],
        n_actions = env2.action_space.n,
        epsilon = BRAIN_GREEDY_EPSILON,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE,
        batch_size = BRAIN_BATCH_SIZE,
        memory_capacity = MEMORY_CAPACITY
    )

    # モデルの構造を定義する。
    brain1.model()
    brain2.model()

    # 損失関数を設定する。
    #brain1.loss()
    #brain2.loss()

    # モデルの最適化アルゴリズムを設定
    brain1.optimizer()
    brain2.optimizer()

    #-----------------------------------
	# Agent の生成
    #-----------------------------------
    agent1 = CartPoleAgent(
        env = env1,
        brain = brain1,
        gamma = BRAIN_GAMMDA
    )

    agent2 = CartPoleAgent(
        env = env2,
        brain = brain2,
        gamma = BRAIN_GAMMDA
    )

    # Agent の Brain を設定
    agent1.set_brain( brain1 )
    agent2.set_brain( brain2 )

    # 学習環境に作成したエージェントを追加
    academy1.add_agent( agent1 )
    academy2.add_agent( agent2 )
    #agent1.print( "after init()" )
    #brain1.print( "after init()" )

    #===================================
    # エピソードの実行
    #===================================
    academy1.academy_run()
    academy2.academy_run()
    #agent1.print( "after run" )
    #brain1.print( "after run" )

    #===================================
    # 学習結果の描写処理
    #===================================
    #academy.save_frames( file_name = "RL_ENV_CartPole-v0_DQN_Episode{}.gif".format(NUM_EPISODE) )

    #---------------------------------------------
    # 利得の履歴の plot
    #---------------------------------------------
    reward_historys1 = agent1.get_reward_historys()
    reward_historys2 = agent2.get_reward_historys()

    plt.clf()
    plt.plot(
        range(0,NUM_EPISODE+1), reward_historys1,
        label = 'with TargetNetwork/ gamma = {}'.format(BRAIN_GAMMDA),
        linestyle = '-',
        linewidth = 1,
        color = 'red'
    )
    plt.plot(
        range(0,NUM_EPISODE+1), reward_historys2,
        label = 'without TargetNetwork/ gamma = {}'.format(BRAIN_GAMMDA),
        linestyle = '--',
        linewidth = 1,
        color = 'blue'
    )
    plt.title( "Reward History" )
    plt.xlim( 0, NUM_EPISODE+1 )
    #plt.ylim( [-0.1, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.legend( loc = "lower right" )
    plt.tight_layout()

    plt.savefig( "{}_DQN2015-DQN2013_Reward_episode{}.png".format( RL_ENV, NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
    plt.show()

    #-----------------------------------
    # 損失関数の plot
    #-----------------------------------
    loss_historys1 = agent1.get_loss_historys()
    loss_historys2 = agent2.get_loss_historys()

    plt.clf()
    plt.plot(
        range( 0, NUM_EPISODE ), loss_historys1,
        label = 'with Target Network / mini_batch_size = %d, learning_rate = %0.4f' % ( BRAIN_BATCH_SIZE, BRAIN_LEARNING_RATE ),
        linestyle = '-',
        linewidth = 1,
        color = 'red'
    )
    plt.plot(
        range( 0, NUM_EPISODE ), loss_historys2,
        label = 'without Target Network / mini_batch_size = %d, learning_rate = %0.4f' % ( BRAIN_BATCH_SIZE, BRAIN_LEARNING_RATE ),
        linestyle = '--',
        linewidth = 1,
        color = 'blue'
    )
    plt.title( "loss / Smooth L1" )
    plt.legend( loc = 'best' )
    plt.xlim( 0, NUM_EPISODE+1 )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()
    plt.savefig( "{}_DQN2015-DQN2013_episode{}.png".format( academy1._env.spec.id, NUM_EPISODE ), dpi = 300, bbox_inches = "tight" )
    plt.show()

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


