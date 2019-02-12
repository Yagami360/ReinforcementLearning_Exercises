# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/02/12] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

# PyTorch
import torch

from collections import namedtuple

Transition = namedtuple(
    typename = "Transition",
    field_names = ( "state", "action", "next_state", "reward" )
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

    def push( self, state, action, next_state, reward ):
        """
        学習用のデータのメモリに、データを push する
        [Args]
            state : <> 現在の状態 s
            action : <> 現在の行動 a
            next_state : <> 次の状態 s'
            reword : <> 報酬        
        [Returns]
        """
        # 現在のメモリサイズが上限値以下なら、新たに容量を確保する。
        if( len(self._memory) < self._capacity ):
            self._memory.append( None )

        # nametuple を使用して、メモリに値を格納
        self._memory[ self._index ] = Transition( state, action, next_state, reward )

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


    def get_mini_batch( self, batch_size ):
        """
        ミニバッチデータを取得する
        [Args]
        [Returns]
        """
        #----------------------------------------------------------------------
        # Experience Replay に基づき、ミニバッチ処理用のデータセットを生成する。
        #----------------------------------------------------------------------
        # メモリサイズがまだミニバッチサイズより小さい場合は、処理を行わない
        if( len(self._memory) < batch_size ):
            return None, None, None, None

        # ミニバッチサイズ以上ならば、学習用データを pop する
        transitions = self.pop( batch_size )
        #print( "transitions :", transitions )

        # 取り出したデータをミニバッチ学習用に reshape
        # transtions : shape = 1 step 毎の (s,a,s',r) * batch_size / shape = 32 * 4
        # → shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = 4 * 32
        batch = batch = Transition( *zip(*transitions) )
        #print( "batch :", batch )

        #
        state_batch = torch.cat( batch.state )
        action_batch = torch.cat( batch.action )
        reward_batch = torch.cat( batch.reward )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        return state_batch, action_batch, reward_batch, non_final_next_states
