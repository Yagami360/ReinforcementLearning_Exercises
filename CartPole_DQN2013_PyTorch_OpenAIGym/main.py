# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from datetime import datetime

# OpenAI Gym
import gym

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch  import nn   # ネットワークの構成関連
import torchvision      # 画像処理関連

# 自作モジュール
from Academy import Academy
from GymAcademy import GymAcademy
from Brain import Brain
from DQN2013MLPBrain import DQN2013MLPBrain
from Agent import Agent
from CartPoleAgent import CartPoleAgent
from ExperienceReplay import ExperienceReplay


#--------------------------------
# 設定可能な定数
#--------------------------------
#DEVICE = "CPU"                     # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                      # 使用デバイス ("CPU" or "GPU")
RL_ENV = "CartPole-v0"              # 利用する強化学習環境の課題名

NUM_EPISODE = 500                   # エピソード試行回数
NUM_TIME_STEP = 200                 # １エピソードの時間ステップの最大数
NUM_SAVE_STEP = 50                  # 強化学習環境の動画の保存間隔（単位：エピソード数）

BRAIN_LEARNING_RATE = 0.0001        # 学習率
BRAIN_BATCH_SIZE = 32               # ミニバッチサイズ (Default:32)
BRAIN_GREEDY_EPSILON_INIT = 0.5     # ε-greedy 法の ε 値の初期値
BRAIN_GREEDY_EPSILON_FINAL = 0.001  # ε-greedy 法の ε 値の最終値
BRAIN_GREEDY_EPSILON_STEPS = 5000   # ε-greedy 法の ε が減少していくフレーム数
BRAIN_GAMMDA = 0.99                 # 利得の割引率
MEMORY_CAPACITY = 10000             # Experience Relay 用の学習用データセットのメモリの最大の長さ


def main():
    """
	強化学習の学習環境用の倒立振子課題 CartPole
    ・エージェントの行動方策の学習ロジックは、DQN (2013年バージョン)
    """
    print("Start main()")
    
    # バージョン確認
    print( "OpenAI Gym", gym.__version__ )
    print( "PyTorch :", torch.__version__ )

    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "開始時間：", datetime.now() )
    print( "DEVICE : ", DEVICE )
    print( "RL_ENV : ", RL_ENV )
    print( "NUM_EPISODE : ", NUM_EPISODE )
    print( "NUM_TIME_STEP : ", NUM_TIME_STEP )
    print( "NUM_SAVE_STEP : ", NUM_SAVE_STEP )
    print( "BRAIN_LEARNING_RATE : ", BRAIN_LEARNING_RATE )
    print( "BRAIN_BATCH_SIZE : ", BRAIN_BATCH_SIZE )
    print( "BRAIN_GREEDY_EPSILON_INIT : ", BRAIN_GREEDY_EPSILON_INIT )
    print( "BRAIN_GREEDY_EPSILON_FINAL : ", BRAIN_GREEDY_EPSILON_FINAL )
    print( "BRAIN_GREEDY_EPSILON_STEPS : ", BRAIN_GREEDY_EPSILON_STEPS )
    print( "BRAIN_GAMMDA : ", BRAIN_GAMMDA )
    print( "MEMORY_CAPACITY : ", MEMORY_CAPACITY )

    #===================================
    # 実行 Device の設定
    #===================================
    if( DEVICE == "GPU" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
    else:
        device = torch.device( "cpu" )

    print( "実行デバイス :", device)
    print( "GPU名 :", torch.cuda.get_device_name(0))
    print( "----------------------------------------------" )

    #===================================
    # 学習環境、エージェント生成フェイズ
    #===================================
    np.random.seed(8)
    random.seed(8)
    torch.manual_seed(8)

    # OpenAI-Gym の ENV を作成
    env = gym.make( RL_ENV )

    # seed 値の設定
    env.seed(8)

    #-----------------------------------
    # Academy の生成
    #-----------------------------------
    academy = GymAcademy( 
        env = env, 
        max_episode = NUM_EPISODE, max_time_step = NUM_TIME_STEP, 
        save_step = NUM_SAVE_STEP
    )

    #-----------------------------------
    # Brain の生成
    #-----------------------------------
    brain = DQN2013MLPBrain(
        device = device,
        n_states = env.observation_space.shape[0],
        n_actions = env.action_space.n,
        epsilon_init = BRAIN_GREEDY_EPSILON_INIT, epsilon_final = BRAIN_GREEDY_EPSILON_FINAL, n_epsilon_step = BRAIN_GREEDY_EPSILON_STEPS,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE,
        batch_size = BRAIN_BATCH_SIZE,
        memory_capacity = MEMORY_CAPACITY
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
    agent = CartPoleAgent(
        env = env,
        brain = brain,
        gamma = BRAIN_GAMMDA,
        max_time_step = NUM_TIME_STEP
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
        linewidth = 0.4,
        color = 'black'
    )
    plt.title( "Reward History" )
    plt.xlim( 0, NUM_EPISODE+1 )
    #plt.ylim( [-0.1, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.legend( loc = "lower right" )
    plt.tight_layout()

    plt.savefig( 
        "{}_DQN2013_Reward_episode{}_lr{}.png".format( RL_ENV, NUM_EPISODE, BRAIN_LEARNING_RATE ), 
        dpi = 300, bbox_inches = "tight"
    )
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
        linewidth = 0.4,
        color = 'black'
    )
    plt.title( "loss / Smooth L1" )
    plt.legend( loc = 'best' )
    plt.xlim( 0, NUM_EPISODE+1 )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "Episode" )
    plt.grid()
    plt.tight_layout()
    plt.savefig( 
        "{}_DQN2013_Loss_episode{}_lr{}.png".format( academy._env.spec.id, NUM_EPISODE, BRAIN_LEARNING_RATE ), 
        dpi = 300, bbox_inches = "tight"
    )
    plt.show()

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


