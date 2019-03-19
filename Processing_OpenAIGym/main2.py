# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt
import os.path

# OpenAI Gym
import gym

# 動画の描写関数用
from matplotlib import animation


# 設定可能な定数
RL_ENV = "Breakout-v0"
NUM_EPISODE = 5          # エピソード試行回数
NUM_TIME_STEP = 100        # １エピソードの時間ステップの最大数


def save_frames( frames, file_name ):
    """
    Displays a list of frames as a gif, with controls
    """
    print( "Start saving frames ..." )
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

    # 動画の保存
    ftitle, fext = os.path.splitext( file_name )
    if( fext == ".gif" ):
        anim.save( file_name, writer = 'imagemagick' )
    else:
        anim.save( file_name )

    plt.close()
    print( "Finish saving frames" )
    return


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
    print( "env.observation_space :", env.observation_space )
    print( "env.action_space :", env.action_space )
    print( "env.unwrapped.get_action_meanings() :", env.unwrapped.get_action_meanings() )     # 行動の値の意味

    # 学習環境の RESET
    # 学習環境を実行する際には、一番最初に RESET して、環境を初期化する必要がある
    # observations : エージェントの状態
    #   Breakout では、画像イメージ 210*160 pixel の RGB 情報（合計：210*160*3=100800個の状態）
    observations = env.reset()
    #print( "observations :", observations )

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
            # done : 終了フラグ / 
            # info : デバッグ情報
            observations, reword, done, info = env.step(action)
            #print( "observations :", observations )
            #print( "reword :", reword )
            #print( "info :", info )

            # 描写
            image = env.render( mode = "rgb_array" )
            frames.append( image )

            # 終了状態ならば、次のエピソードを実行し、学習環境を RESET
            if( done == True ):
                break

    #===================================
    # 学習結果の描写処理
    #===================================
    save_frames( 
        frames, 
        "RL_ENV_{}_Episode{}.gif".format(env.spec.id, NUM_EPISODE )
    )


    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()


