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
from Brain import Brain


class MazeBrain( Brain ):
    """
    迷宮問題の Brain
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける


    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self ):
        super().__init__()
        self._action = ["Up", "Right", "Down", "Left"]
        return
