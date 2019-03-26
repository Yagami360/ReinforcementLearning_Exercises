# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from datetime import datetime

# OpenAI Gym
import gym

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch  import nn   # ネットワークの構成関連
import torchvision      # 画像処理関連

# OpenCV
import cv2

# 自作モジュール
from Academy import Academy
from GymAcademy import GymAcademy
from Brain import Brain
from DQN2015CNNBrain import DQN2015CNNBrain
from Agent import Agent
from AtariAgent import AtariAgent
from ExperienceReplay import ExperienceReplay

from AtariWrappers import *


#--------------------------------
# 設定可能な定数
#--------------------------------
#DEVICE = "CPU"                         # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                          # 使用デバイス ("CPU" or "GPU")

#RL_ENV = "Breakout-v0"                 # 利用する強化学習環境の課題名
RL_ENV = "BreakoutNoFrameskip-v0"      
#RL_ENV = "BreakoutNoFrameskip-v4"      
#RL_ENV = "PongNoFrameskip-v0"

NUM_EPISODE = 5000                      # エピソード試行回数 (Default:10000)
NUM_TIME_STEP = 1000                    # １エピソードの時間ステップの最大数
NUM_SAVE_STEP = 100                     # 強化学習環境の動画の保存間隔（単位：エピソード数）

NUM_NOOP = 30                           # エピソード開始からの何も学習しないステップ数 (Default:30)
NUM_SKIP_FRAME = 4                      # スキップするフレーム数 (Default:4)
NUM_STACK_FRAME = 4                     # モデルに一度に入力する画像データのフレーム数 (Default:4)

BRAIN_LEARNING_RATE = 0.00005           # 学習率 (Default:5e-5)
BRAIN_BATCH_SIZE = 32                   # ミニバッチサイズ (Default:32)
BRAIN_GREEDY_EPSILON_INIT = 1.0         # ε-greedy 法の ε 値の初期値 (Default:1.0)
BRAIN_GREEDY_EPSILON_FINAL = 0.01       # ε-greedy 法の ε 値の最終値 (Default:0.1)
BRAIN_GREEDY_EPSILON_STEPS = 50000      # ε-greedy 法の ε が減少していくフレーム数 (Default:1_000_000)
BRAIN_GAMMDA = 0.99                     # 利得の割引率 (Default:0.99)
BRAIN_FREC_TARGET_UPDATE = 1000         # Target Network との同期頻度（Default:10_000） 
MEMORY_CAPACITY = 10000                 # Experience Relay 用の学習用データセットのメモリの最大の長さ (Default:1_000_000)


def main():
    """
	強化学習の学習環境用のブロック崩しゲーム（Breakout）
    ・エージェントの行動方策の学習ロジックは、DQN (2015年Natureバージョン)
    """
    print("Start main()")
    
    # バージョン確認
    print( "OpenAI Gym", gym.__version__ )
    print( "PyTorch :", torch.__version__ )
    print( "OpenCV :", cv2.__version__ )

    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "開始時間：", datetime.now() )
    print( "DEVICE : ", DEVICE )
    print( "RL_ENV : ", RL_ENV )
    print( "NUM_EPISODE : ", NUM_EPISODE )
    print( "NUM_TIME_STEP : ", NUM_TIME_STEP )
    print( "NUM_SAVE_STEP : ", NUM_SAVE_STEP )
    print( "NUM_NOOP : ", NUM_NOOP )
    print( "NUM_SKIP_FRAME : ", NUM_SKIP_FRAME )
    print( "NUM_STACK_FRAME : ", NUM_STACK_FRAME )
    print( "BRAIN_LEARNING_RATE : ", BRAIN_LEARNING_RATE )
    print( "BRAIN_BATCH_SIZE : ", BRAIN_BATCH_SIZE )
    print( "BRAIN_GREEDY_EPSILON_INIT : ", BRAIN_GREEDY_EPSILON_INIT )
    print( "BRAIN_GREEDY_EPSILON_FINAL : ", BRAIN_GREEDY_EPSILON_FINAL )
    print( "BRAIN_GREEDY_EPSILON_STEPS : ", BRAIN_GREEDY_EPSILON_STEPS )
    print( "BRAIN_GAMMDA : ", BRAIN_GAMMDA )
    print( "BRAIN_FREC_TARGET_UPDATE : ", BRAIN_FREC_TARGET_UPDATE )
    print( "MEMORY_CAPACITY : ", MEMORY_CAPACITY )
    print( "----------------------------------------------" )

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

    #===================================
    # 学習環境、エージェント生成フェイズ
    #===================================
    # seed 値の設定
    np.random.seed(8)
    random.seed(8)
    torch.manual_seed(8)

    # OpenAI-Gym の ENV を作成
    env = make_env( 
        device = device,
        env_id = RL_ENV, 
        seed = 8,
        n_noop_max = NUM_NOOP,
        n_skip_frame = NUM_SKIP_FRAME,
        n_stack_frames = NUM_STACK_FRAME
    )

    print( "env.observation_space :", env.observation_space )
    print( "env.action_space :", env.action_space )
    print( "env.unwrapped.get_action_meanings() :", env.unwrapped.get_action_meanings() )     # 行動の値の意味

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
    brain = DQN2015CNNBrain(
        device = device,
        n_states = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2],
        n_actions = env.action_space.n,
        epsilon_init = BRAIN_GREEDY_EPSILON_INIT, epsilon_final = BRAIN_GREEDY_EPSILON_FINAL, n_epsilon_step = BRAIN_GREEDY_EPSILON_STEPS,
        gamma = BRAIN_GAMMDA,
        learning_rate = BRAIN_LEARNING_RATE,
        batch_size = BRAIN_BATCH_SIZE,
        memory_capacity = MEMORY_CAPACITY,
        n_stack_frames = NUM_STACK_FRAME,
        n_skip_frames = NUM_SKIP_FRAME,
        n_frec_target_update = BRAIN_FREC_TARGET_UPDATE
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
    agent = AtariAgent(
        device = device,
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
        linewidth = 0.2,
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
        "{}_DQN2015_Reward_episode{}_ts{}_lr{}_noop{}.png".format( RL_ENV, NUM_EPISODE, NUM_TIME_STEP, BRAIN_LEARNING_RATE, NUM_NOOP ), 
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
        linewidth = 0.2,
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
        "{}_DQN2015_Loss_episode{}_ts{}_lr{}_noop{}.png".format( RL_ENV, NUM_EPISODE, NUM_TIME_STEP, BRAIN_LEARNING_RATE, NUM_NOOP ),  
        dpi = 300, bbox_inches = "tight"
    )
    plt.show()

    print( "終了時間：", datetime.now() )
    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


