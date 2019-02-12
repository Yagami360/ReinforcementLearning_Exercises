# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/02/12] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

from collections import namedtuple

Transition = namedtuple(
    typename = "Transition",
    field_names = ( "state", "action", "next_state", "reword" )
)


class ExperienceReplay( object ):
    """
    Experience Replay による学習用のミニバッチデータの生成を行うクラス

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _capacity : [int] メモリの最大値
        _memory : [list] (s,a,s',a',r) のリスト（学習用データ）
        _index : [int] 現在のメモリのインデックス
             
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        capacity = 10000
    ):
        self._capacity = capacity
        self._memory = []
        self._index = 0
        return

    def __len__( self ):
        return len( self._memory )

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleAgent" )
        print( self )
        print( str )
        print( "_capacity :", self._capacity )
        print( "_memory : \n", self._memory )
        print( "_index : \n", self._index )
        return

    def push( self, state, action, next_state, reword ):
        """
        学習用のデータのメモリに、データを push する
        [Args]
            state : <list> 現在の状態 s
            action : <int> 現在の行動 a
            next_state : <int> 次の状態 s'
            reword : <float> 報酬        
        [Returns]
        """
        # 現在のメモリサイズが上限値以下なら、新たに容量を確保する。
        if( len(self._memory) < self._capacity ):
            self._memory.append( None )

        # nametuple を使用して、メモリに値を格納
        self._memory[ self._index ] = Transition( state, action, next_state, reword )

        # 現在のインデックスをづらす
        self._index = ( self._index + 1 ) % self._capacity

        return

    def pop( self, batch_size ):
        """
        ミニバッチサイズ分だけ、ランダムにメモリの内容を取り出す
        [Args]
        [Returns]
            
        """
        return random.sample( self._memory, batch_size )

    def create_memory( self, batch_size ):
        """
        Experience Replay に基づき、ミニバッチ処理用のデータセットを生成する。
        [Args]
        [Returns]
        """
        #------------------------------------------------
        # ミニバッチ処理用のデータセットの作成
        #------------------------------------------------
        transtions = self.pop( batch_size )

        return

