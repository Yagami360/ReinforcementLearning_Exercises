# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/07] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Agent import Agent


class Academy( object ):
    """
    エージェントの強化学習環境
    ・強化学習モデルにおける環境 Enviroment に対応
    ・学習や推論を行うための設定を行う。
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _max_episode : int
                       エピソードの最大回数
                       最大回数数に到達すると、Academy と全 Agent のエピソードを完了する。
        _max_time_step : int
                        時間ステップの最大回数

        _agents : list<AgentBase>

        _done : bool
            エピソードが完了したかのフラグ

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, max_episode = 1, max_time_step = 100 ):
        self._max_episode = max_episode
        self._max_time_step = max_time_step
        self._agents = []
        self._done = False
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

    def Done( self ):
        """
        エピソードを完了にする。
        """
        self._done = True
        return

    def IsDone( self ):
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
        """
        # エピソードを試行
        for episode in range( 0, self._max_episode ):
            # 学習環境を RESET
            self.academy_reset()

            # 時間ステップを 1ステップづつ進める
            for time_step in range( 0 ,self._max_time_step ):
                for agent in self._agents:
                    agent.agent_step( episode, time_step )

            # Academy と全 Agents のエピソードを完了
            self._done = True
            for agent in self._agents:
                agent.agent_on_done()

        return
