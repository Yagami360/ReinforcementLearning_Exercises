# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# 自作クラス
from Agent import Agent


class CartPoleAgent( Agent ):
    """
    """
    def __init__( 
        self, 
        brain = None, 
        gamma = 0.9
    ):
        super().__init__( brain, gamma, 0 )
        self._q_function_historys = []
        self._v_function_historys = []
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleAgent" )
        print( self )
        print( str )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_reword : \n", self._reword )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_state : \n", self._state )
        print( "_action : \n", self._action )
        print( "_s_a_historys : \n", self._s_a_historys )
        #print( "_q_function_historys : \n", self._q_function_historys )
        #print( "_v_function_historys : \n", self._v_function_historys )
        print( "----------------------------------" )
        return


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._reword = 0.0
        self._done = False
        return


    def agent_step( self, episode, time_step ):
        """
        エージェント [Agent] の次の状態を決定する。
        ・Academy から各時間ステップ度にコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
            time_step : 現在の時間ステップ

        [Returns]
            done : bool
                   エピソードの完了フラグ
        """
        done = False

        print( "現在のエピソード数：", episode )
        print( "現在の時間ステップ数：", time_step )

        # 離散化した現在の状態 s_t を求める
        state = self._brain.digitize_state( self._observations )
        print( "state :", state )

        # 離散化した現在の状態 s_t を元に、行動 a_t を求める

        # 行動の実行により、次の時間での状態 s_{t+1} と報酬 r_{t+1} を求める。

        return done
