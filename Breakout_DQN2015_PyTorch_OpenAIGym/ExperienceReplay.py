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
from collections import deque

Transition = namedtuple(
    typename = "Transition",
    field_names = ( "state", "action", "next_state", "reward", "done" )
)


class ExperienceReplay( object ):
    """
    Experience Replay による学習用のミニバッチデータの生成を行うクラス

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <torch.device> 実行デバイス
        _capacity : [int] メモリの最大値
        _memory : [list] (s,a,s',a',r) のリスト（学習用データ）
             
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device,
        capacity = 1000
    ):
        self._device = device
        self._capacity = capacity
        self._memory = deque( maxlen = capacity )
        return

    def __len__( self ):
        return len( self._memory )

    def print( self, str ):
        print( "----------------------------------" )
        print( "ExperienceReplay" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_capacity :", self._capacity )
        print( "_memory : \n", self._memory )
        return

    def push( self, state, action, next_state, reward, done ):
        """
        学習用のデータのメモリに、データを push する
        [Args]
            state : <ndarry> 現在の状態 s
            action : <int> 現在の行動 a
            next_state : <ndarray> 次の状態 s'
            reword : <float> 報酬        
            done : <bool> 完了フラグ
        [Returns]
        """
        # nametuple を使用して、メモリに値を格納
        self._memory.append( Transition( state, action, next_state, reward, done ) )
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
            state_batch : <Tensor/torch.float32> 現在の状態 s のミニバッチデータ / shape = [batch_size, n_channels, width,height]
            action_batch : <Tensor/torch.int64> 現在の行動 a のミニバッチデータ / shape = [batch_size,1]
            next_state_batch : <Tensor/torch.float32> 次の状態 s' のミニバッチデータ / shape = [batch_size, n_channels, width,height]
            reword_batch : <Tensor/torch.float32> 報酬のミニバッチデータ / shape = [batch_size,1]
            done_batch : <Tensor/torch.float32> 完了フラグのミニバッチデータ / shape = [batch_size,1]
        """
        #----------------------------------------------------------------------
        # Experience Replay に基づき、ミニバッチ処理用のデータセットを生成する。
        #----------------------------------------------------------------------
        # メモリサイズがまだミニバッチサイズより小さい場合は、処理を行わない
        if( len(self._memory) < batch_size ):
            return None, None, None, None, None

        # ミニバッチサイズ以上ならば、学習用データを pop する
        transitions = self.pop( batch_size )
        #print( "transitions :", transitions )

        # 取り出したデータをミニバッチ学習用に reshape
        # transtions : shape = 1 step 毎の (s,a,s',r) * batch_size / shape = 32 * 4
        # → shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = 4 * 32
        batch = Transition( *zip(*transitions) )
        #print( "batch :", batch )

        #
        state_batch = torch.cat( batch.state ).to(self._device)
        action_batch = torch.cat( batch.action ).to(self._device)
        reward_batch = torch.cat( batch.reward ).to(self._device)
        done_batch = torch.cat( batch.done ).to(self._device)

        return batch, state_batch, action_batch, reward_batch, done_batch
