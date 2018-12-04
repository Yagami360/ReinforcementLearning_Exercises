# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

import numpy as np
import matplotlib.pyplot as plt

# 自作モジュール
from Academy import Academy
from Brain import Brain
from AgentBase import AgentBase
from MazeAgent import MazeAgent


def main():
    """
	強化学習のトイプロブレム用の迷路探索問題
	・単純な迷路探索問題を、Unity ML-Agents のフレームワークを参考にして実装
    """
    print("Start main()")
    
	# 
    agent = MazeAgent()
    agent.print( "after init()" )

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()
    