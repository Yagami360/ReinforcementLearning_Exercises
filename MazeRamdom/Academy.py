# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/05] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from AgentBase import AgentBase


class Academy( object ):
    """
    エージェントの学習環境
    ・
    [protected]
        _max_step : int
                    学習環境のシミュレーションの最大回数
        _agents : list<AgentBase>

    """
    def __init__( self, max_step = 1000 ):
        self._max_step = max_step
        self._agents = []
        return

    def academy_reset( self ):
        """
        学習環境をリセットする。
        """
        if( self._agents != None ):
            for agent in self._agents:
                agent.agent_reset()        

        return

    def add_agent( self, agent ):
        """
        学習環境にエージェントを追加する。
        """
        self._agents.append( agent )
        return

    def academy_step( self ):
        """
        次のシミュレーションステップ
        """
        for step in range( 1,self._max_step ):
            for agent in self._agents:
                done = agent.agent_step( step )

                if ( done == True ):
                    break
            if ( done == True ):
                break

        return

    def draw_academy( self ):
        """
        エージェントの学習環境を描写する。
        """

        return