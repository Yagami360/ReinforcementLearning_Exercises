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


    def push_tensor( self, state, action, next_state, reward, done ):
        """
        学習用のデータのメモリに、データを Tensor 型として push する
        [Args]
            state : <ndarry> 現在の状態 s
            action : <int> 現在の行動 a
            next_state : <ndarray> 次の状態 s'
            reword : <float> 報酬        
            done : <bool> 完了フラグ
        [Returns]
        """
        #-----------------------------------------
        # メモリに保管する値を Tensor に変換
        # この際に、ミニバッチ用の次元を追加
        #-----------------------------------------
        state = torch.from_numpy( state ).to(self._device)
        state = torch.unsqueeze( state, dim = 0 ).to(self._device)
        action = torch.LongTensor( [[action]] ).to(self._device)                # [[x]] で shape = [1,1] にしておき、ミニバッチ用の次元を用意
        reward = torch.FloatTensor( [[reward]] ).to(self._device)
        next_state = torch.from_numpy( next_state ).to(self._device)
        next_state = torch.unsqueeze( next_state, dim = 0 ).to(self._device)
        done = torch.FloatTensor( [[done]] ).to(self._device)

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
        Experience Replay に基いた、ミニバッチデータを取得する
        ※ メモリの値を Tensor ではなく、nump で保管している場合の処理
        ※ メモリを Tensor で保管しているのに比べて、処理は遅くなるが、GPU メモリの使用量は少ない

        [Args]
            batch_size : <int> ミニバッチサイズ
        [Returns]
            state_batch : <Tensor/torch.float32> 現在の状態 s のミニバッチデータ / shape = [batch_size, n_channels, width,height]
            action_batch : <Tensor/torch.int64> 現在の行動 a のミニバッチデータ / shape = [batch_size,1]
            next_state_batch : <Tensor/torch.float32> 次の状態 s' のミニバッチデータ / shape = [batch_size, n_channels, width,height]
            reword_batch : <Tensor/torch.float32> 報酬のミニバッチデータ / shape = [batch_size,1]
            done_batch : <Tensor/torch.float32> 完了フラグのミニバッチデータ / shape = [batch_size,1]
        """
        # データを pop する
        # batch : shape = 1 step 毎の (s,a,s',r,done)のペア * batch_size / shape = 32 * 5
        batch = self.pop( batch_size )
        #print( "batch :", batch )

        #--------------------------------------------------------------------
        # numpy → Tensor に変換
        # ミニバッチデータの段階で Tensor に変換するのは、GPUメモリ削減のため
        # .to(self._device) で Tensor を GPU に転送
        #--------------------------------------------------------------------
        # 以下の操作を一度に行っている。
        # state_batch = torch.from_numpy( b.state ).float().to(self._device)
        # state_batch = torch.unsqueeze( state_batch, dim = 0 ).to(self._device)
        # state_batch = torch.cat( batch_state_tsr ).to(self._device)
        state_batch = torch.cat(
            [ torch.from_numpy(b.state).unsqueeze(0) for b in batch ], 
            dim=0
        ).to(self._device)

        action_batch = torch.cat(
            [ torch.LongTensor([[b.action]]) for b in batch ]   # [[x]] で shape = [batch_size, 1] にしておく
        ).to(self._device)

        reward_batch = torch.cat(
            [ torch.FloatTensor([[b.reward]]) for b in batch ]
        ).to(self._device)

        next_state_batch = torch.cat(
            [ torch.from_numpy(b.next_state).unsqueeze(0) for b in batch ], 
            dim=0
        ).to(self._device)

        done_batch = torch.cat(
            [ torch.FloatTensor([[b.done]]) for b in batch ]
        ).to(self._device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def get_mini_batch_from_tensor( self, batch_size ):
        """
        Experience Replay に基いた、ミニバッチデータを取得する。
        ※ メモリの値を Tensor で保管しているときの処理
        ※ メモリを非Tensor で保管しているのに比べて、処理は早くなるが、GPU メモリの使用量は多い

        [Args]
            batch_size : <int> ミニバッチサイズ
        [Returns]
            state_batch : <Tensor/torch.float32> 現在の状態 s のミニバッチデータ / shape = [batch_size, n_channels, width,height]
            action_batch : <Tensor/torch.int64> 現在の行動 a のミニバッチデータ / shape = [batch_size,1]
            next_state_batch : <Tensor/torch.float32> 次の状態 s' のミニバッチデータ / shape = [batch_size, n_channels, width,height]
            reword_batch : <Tensor/torch.float32> 報酬のミニバッチデータ / shape = [batch_size,1]
            done_batch : <Tensor/torch.float32> 完了フラグのミニバッチデータ / shape = [batch_size,1]
        """
        # batch : shape = 1 step 毎の (s,a,s',r,done)のペア * batch_size / shape = 32 * 5
        batch = self.pop( batch_size )
        #print( "batch :", batch )

        # batch : shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = [5,32] のタプル
        batch = Transition( *zip(*batch) )

        # torch.cat() : Tensorをリスト入れてして渡すことで、それらを連結したTensorを返す。連結する軸はdimによって指定。
        #               Tensor の配列(shepe=batch_size)である batch[0]=batch.state → 1つの Tensor である state.batch に変換
        state_batch = torch.cat( batch.state )
        action_batch = torch.cat( batch.action )
        reward_batch = torch.cat( batch.reward )
        next_state_batch = torch.cat( batch.next_state )
        done_batch = torch.cat( batch.done )

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
