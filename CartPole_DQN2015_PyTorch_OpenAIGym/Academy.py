# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/07] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 自作クラス
from Agent import Agent


class Academy( object ):
    """
    エージェントの強化学習環境
    ・強化学習モデルにおける環境 Enviroment に対応
    ・学習や推論を行うための設定を行う。
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _max_episode : <int> エピソードの最大回数。最大回数数に到達すると、Academy と全 Agent のエピソードを完了する。
        _max_time_step : <int> 時間ステップの最大回数
        _save_step : <int> 保存間隔（エピソード数）
        _agents : list<AgentBase>
        _done : <bool> エピソードが完了したかのフラグ
        _frames : list <ndarray>　動画のフレーム（１つの要素が１画像のフレーム）

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, max_episode = 1, max_time_step = 100, save_step = 5 ):
        self._max_episode = max_episode
        self._max_time_step = max_time_step
        self._save_step = save_step
        self._agents = []
        self._done = False
        self._frames = []
        return

    def academy_reset( self ):
        """
        学習環境をリセットする。
        """
        if( self._agents != None ):
            for agent in self._agents:
                agent.agent_reset()        

        self._done = False
        return

    def done( self ):
        """
        エピソードを完了にする。
        """
        self._done = True
        return

    def is_done( self ):
        """
        Academy がエピソードを完了したかの取得
        """
        return self._done

    def add_agent( self, agent ):
        """
        学習環境にエージェントを追加する。
        """
        self._agents.append( agent )
        return


    def academy_run( self ):
        """
        学習環境を実行する
        """
        # エピソードを試行
        for episode in tqdm( range( 0, self._max_episode ), desc = "Episode" ):
            # 学習環境を RESET
            self.academy_reset()

            # 時間ステップを 1ステップづつ進める
            for time_step in range( 0 ,self._max_time_step ):
                dones = []

                # 学習環境の動画のフレームを追加
                if( episode % self._save_step == 0 ):
                    self.add_frame( episode, time_step )
                if( episode == self._max_episode - 1 ):
                    self.add_frame( episode, time_step )

                for agent in self._agents:
                    done = agent.agent_step( episode, time_step )
                    dones.append( done )

                # 全エージェントが完了した場合
                if( all(dones) == True ):
                    break

            # Academy と全 Agents のエピソードを完了
            self._done = True
            for agent in self._agents:
                agent.agent_on_done( episode, time_step )

            # 動画を保存
            if( episode % self._save_step == 0 ):
                self.save_frames( "RL_ENV_Episode{}.gif".format(episode) )
                self._frames = []

            if( episode == self._max_episode - 1 ):
                self.save_frames( "RL_ENV_Episode{}.gif".format(episode) )
                self._frames = []
        
        return


    def add_frame( self, episode, times_step, total_times_step ):
        """
        強化学習環境の１フレームを追加する
        [Args]
            episode : <int> 現在のエピソード数
            time_step : <int> 現在のエピソードにおける経過時間ステップ数
            total_time_step : <int> 全てのエピソードにおける全経過時間ステップ数
        """
        #frame = None
        #self._frames.append( frame )
        return

    def save_frames( self, file_name ):
        """
        外部ファイルに動画を保存する。
        """
        return

