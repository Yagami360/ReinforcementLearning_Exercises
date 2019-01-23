# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt

# OpenAI Gym
import gym

# 動画の描写関数用
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
#from IPython.display import display


# 設定可能な定数
RL_ENV = "CartPole-v0"     # 利用する強化学習環境の課題名
#RL_ENV = "MountainCar-v0"     # 利用する強化学習環境の課題名
NUM_EPISODE = 5          # エピソード試行回数
NUM_TIME_STEP = 100        # １エピソードの時間ステップの最大数


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(
        figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
        dpi=72
    )
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
               plt.gcf(), 
               animate, 
               frames=len(frames),
               interval=50
    )

    anim.save( 'RL_ENV_{}.mp4'.format( RL_ENV ) )  # 追記：動画の保存です
    #display( display_animation(anim, default_mode='loop') )


def main():
    """
	OpenAI Gym での処理フローの練習コード
    """
    print("Start main()")

    # バージョン確認
    print( "OpenAI Gym", gym.__version__ )

    #===================================
    # 学習環境の生成フェイズ
    #===================================
    env = gym.make( RL_ENV )
    print( "env :", env )

    # 学習環境の RESET
    # 学習環境を実行する際には、一番最初に RESET して、環境を初期化する必要がある
    # observations : エージェントの状態
    #   CartPole では、カートの棒の状態を表す４つの変数
    #   observation[0] : カートの位置 / -2.4 ~ 2.4
    #   observation[1] : カートの速度 / -inf ~ inf
    #   observation[2] : 棒の速度 / -41.8° ~ 41.8°
    #   observation[3] : 棒の角速度 / -inf ~ inf
    observations = env.reset()
    print( "observations :", observations )

    #===================================
    # エピソードの実行
    #===================================
    # 動画のフレーム
    frames = []

    for episode in range(0, NUM_EPISODE):
        print( "---------------------------------------------")
        print( "エピソードの開始")
        print( "---------------------------------------------")
        print( "episode :", episode )
        
        # エピソードの開始の度に、学習環境を RESET
        observations = env.reset()

        for step in range(0, NUM_TIME_STEP):
            print( "step :", step )

            # ランダムな行動
            # CartPoleでは左（0）、右（1）の2つの行動だけなので、actionの値は0か1になります。
            action = env.action_space.sample()
            print( "action :", action )
            
            # reword : 即時報酬
            #          CartPole では、action 実行後に、
            #          1 : -カートの位置 = ±2.4以内 && 棒の角度 = 20.9°以内
            #          0 : その他 
            # done : 終了フラグ / 
            # info : デバッグ情報
            observations, reword, done, info = env.step(action)
            print( "observations :", observations )
            print( "reword :", reword )
            print( "info :", info )

            # 描写
            image = env.render( mode = "rgb_array" )
            frames.append( image )

            # 終了状態ならば、次のエピソードを実行し、学習環境を RESET
            if( done == True ):
                break

    #===================================
    # 学習結果の描写処理
    #===================================
    display_frames_as_gif( frames )


    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


