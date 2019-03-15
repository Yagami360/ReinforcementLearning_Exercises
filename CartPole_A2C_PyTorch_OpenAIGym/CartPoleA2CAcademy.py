# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/13] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os.path

# 自作クラス
from Academy import Academy
from CartPoleAcademy import CartPoleAcademy
from Agent import Agent


class CartPoleA2CAcademy( CartPoleAcademy ):
    """
    エージェントの強化学習環境
    ・強化学習モデルにおける環境 Enviroment に対応
    ・学習や推論を行うための設定を行う。
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _env : OpenAIGym の ENV

        _frames : list<>
            動画のフレーム（１つの要素が１画像のフレーム）
        
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        env, 
        max_epoches = 1, k_step = 5,
        save_step = 100
    ):
        super().__init__( env, -1, -1, save_step )
        self._max_epoches = max_epoches
        self._k_step = k_step
        return


    def academy_reset( self ):
        """
        学習環境をリセットする。
        ・エピソードの開始時にコールされる
        """
        if( self._agents != None ):
            for agent in self._agents:
                agent.agent_reset()        

        self._done = False
        #self._env.reset()
        return

    def academy_run( self ):
        """
        学習環境を実行する
        """
        self.academy_reset()

        episode = 0     # 現在のエピソード数
        time_step = 0   # 現在の時間ステップ t

        # 繰り返し試行
        for epoch in range( 0, self._max_epoches ):
            # k step
            for kstep in range( self._k_step ):
                dones = []
                for agent in self._agents:
                    done = agent.agent_step( episode, time_step )
                    dones.append( done )

                time_step += 1

                # 学習環境の動画のフレームを追加
                if( episode % self._save_step == 0 ):
                    self.add_frame( episode, time_step )

                # 全エージェントが完了した場合
                if( all(dones) == True ):
                    self._done = True
                    #break

            # k_step 完了後の処理
            for agent in self._agents:
                agent.agent_on_kstep_done( episode, time_step )

            # 学習環境を RESET
            if ( self._done == True ):
                print( "epoch :", epoch )
                # Academy と全 Agents のエピソードを完了
                for agent in self._agents:
                    agent.agent_on_done( episode, time_step )
                
                # 動画を保存
                if( episode % self._save_step == 0 ):
                    self.save_frames( "RL_ENV_{}_Episode{}.gif".format(self._env.spec.id, episode) )
                    self._frames = []

                time_step = 0
                episode += 1
                self.academy_reset()

        return

