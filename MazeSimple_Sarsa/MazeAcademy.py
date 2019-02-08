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
from Academy import Academy
from Agent import Agent


class MazeAcademy( Academy ):
    """
    迷宮問題のエージェントの学習環境
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
    
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, max_episode = 1, max_time_step = 100 ):
        super().__init__( max_episode )
        #self._frames = []
        return

