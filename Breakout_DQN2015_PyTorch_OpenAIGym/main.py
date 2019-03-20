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
from BreakoutAcademy import BreakoutAcademy
from Brain import Brain
from BreakoutDQN2015Brain import BreakoutDQN2015Brain
from Agent import Agent
from BreakoutAgent import BreakoutAgent
from ExperienceReplay import ExperienceReplay

from BreakoutAtariWrappers import *


#--------------------------------
# 設定可能な定数
#--------------------------------
RL_ENV = "BreakoutNoFrameskip-v0"   # 利用する強化学習環境の課題名
NUM_EPISODE = 200                   # エピソード試行回数
NUM_TIME_STEP = 1000                # １エピソードの時間ステップの最大数
NUM_NOOP = 30                       # エピソード開始からの何も学習しないステップ数
NUM_SKIP_FRAME = 4                  # スキップするフレーム数
NUM_STACK_FRAME = 1                 # モデルに一度に入力する画像データのフレーム数
BRAIN_LEARNING_RATE = 0.0001        # 学習率
BRAIN_BATCH_SIZE = 32               # ミニバッチサイズ
BRAIN_GREEDY_EPSILON = 0.5          # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99                 # 利得の割引率
MEMORY_CAPACITY = 10000             # Experience Relay 用の学習用データセットのメモリの最大の長さ


def main():
    """
	強化学習の学習環境用のブロック崩しゲーム（Breakout）
    ・エージェントの行動方策の学習ロジックは、DQN (2015年Natureバージョン)
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
    #env = gym.make( RL_ENV )
    #env.seed(8)
    env = make_env( 
        env_id = RL_ENV, 
        seed = 8,
        n_noop_max = NUM_NOOP,
        n_skip_frame = NUM_SKIP_FRAME
    )

    print( "env.observation_space :", env.observation_space )
    print( "env.action_space :", env.action_space )
    print( "env.unwrapped.get_action_meanings() :", env.unwrapped.get_action_meanings() )     # 行動の値の意味

    #-----------------------------------
    # Academy の生成
    #-----------------------------------
    academy = BreakoutAcademy( 
        env = env, 
        max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP, 
        save_step = 25
    )

    #-----------------------------------
    # Brain の生成
    #-----------------------------------
    brain = BreakoutDQN2015Brain(
        n_states = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2],
        n_actions = env.action_space.n,
        epsilon = BRAIN_GREEDY_EPSILON,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE,
        batch_size = BRAIN_BATCH_SIZE,
        memory_capacity = MEMORY_CAPACITY,
        n_stack_frames = NUM_STACK_FRAME
    )
    
    # モデルの構造を定義する。
    #brain.model()

    # 損失関数を設定する。
    #brain.loss()

    # モデルの最適化アルゴリズムを設定
    brain.optimizer()

    #-----------------------------------
	# Agent の生成
    #-----------------------------------
    agent = BreakoutAgent(
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

    plt.savefig( "{}_DQN2015_Reward_episode{}.png".format( RL_ENV, NUM_EPISODE), dpi = 300, bbox_inches = "tight" )
    plt.show()

    #-----------------------------------
    # 損失関数の plot
    #-----------------------------------
    loss_historys = agent.get_loss_historys()

    plt.clf()
    plt.plot(
        range( 0, NUM_EPISODE ), loss_historys,
        label = 'mini_batch_size = %d, learning_rate = %0.4f' % ( BRAIN_BATCH_SIZE, BRAIN_LEARNING_RATE ),
        linestyle = '-',
        #linewidth = 2,
        color = 'black'
    )
    plt.title( "loss / Smooth L1" )
    plt.legend( loc = 'best' )
    plt.xlim( 0, NUM_EPISODE+1 )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()
    plt.savefig( "{}_DQN2015_episode{}.png".format( academy._env.spec.id, NUM_EPISODE ), dpi = 300, bbox_inches = "tight" )
    plt.show()

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


