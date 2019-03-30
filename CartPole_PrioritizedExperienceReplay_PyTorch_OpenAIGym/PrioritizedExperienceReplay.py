# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/07] : 新規作成
    [19/03/29] : データ構造を list から deque に変更
                 done フラグもバッファに保管するように変更
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


class PrioritizedExperienceReplay( object ):
    """
    Prioritized　Experience Replay による学習用のミニバッチデータの生成を行うクラス
    ・サンプリング優先度は、TD誤差で判断する。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける        
        _device : <torch.device> 実行デバイス
        _capacity : [int] メモリの最大値
        _memory : [list] (s,a,s',a',r) のリスト（学習用データ）
        _memory_td_error : [list] サンプリング優先度を表すTD誤差のメモリ
        _td_error_epsilon : [float] サンプリング優先度を表すTD誤差のバイアス値

        _change_episode : Experience Replay → PrioritizedExperienceReplay に切り替えるエピソード数

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device,
        capacity = 10000,
        td_error_epsilon = 0.0001,
        change_episode = 30
    ):
        self._device = device
        self._capacity = capacity
        self._memory = deque( maxlen = capacity )
        self._memory_td_error = deque( maxlen = capacity )
        self._td_error_epsilon = td_error_epsilon
        self._change_episode = change_episode
        return

    def __len__( self ):
        return len( self._memory )

    def print( self, str ):
        print( "----------------------------------" )
        print( "PrioritizedExperienceReplay" )
        print( self )
        print( str )
        print( "_capacity :", self._capacity )
        print( "_memory :\n", self._memory )
        print( "_memory_td_error :\n", self._memory_td_error )
        print( "_td_error_epsilon : ", self._td_error_epsilon )
        print( "_change_episode : ", self._change_episode )
        return

    def get_memory( self ):
        return self._memory

    def set_td_error_memory( self, td_errors ):
        self._memory_td_error = td_errors

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
        state = torch.from_numpy( state ).type(torch.FloatTensor).to(self._device)
        state = torch.unsqueeze( state, dim = 0 ).to(self._device)
        action = torch.LongTensor( [[action]] ).to(self._device)                # [[x]] で shape = [1,1] にしておき、ミニバッチ用の次元を用意
        reward = torch.FloatTensor( [reward] ).to(self._device)
        next_state = torch.from_numpy( next_state ).type(torch.FloatTensor).to(self._device)
        next_state = torch.unsqueeze( next_state, dim = 0 ).to(self._device)
        done = torch.FloatTensor( [done] ).to(self._device)

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

    def push_td_error( self, td_error ):
        """
        サンプリング優先度を表すTD誤差をメモリに格納する。

        [Args]
            td_error : [float] TD誤差
        """
        # メモリに値を格納
        self._memory_td_error.append( td_error )
        return


    def get_prioritized_indexes( self, batch_size ):
        """
        TD誤差に基づき、優先付けされたサンプリング対象の index のリストを求める。
        ・サンプリングの確率分布は Stochastic sampling method で、優先度は Proportional Prioritization ( direct ) で実装

        [Returns]
            indexes : list<int> TD誤差に基づき、優先付けされたサンプリング対象の index のリスト
        """
        # TD誤差の絶対値の和を計算
        sum_abs_td_error = np.sum( np.absolute(self._memory_td_error) )

        # TD誤差のバイアス値を加算（Proportional Prioritization）
        sum_abs_td_error += self._td_error_epsilon

        # batch_size分の 0 ~ sum_abs_td_error の値の範囲での乱数を生成して、昇順に並べる
        rand_list = np.random.uniform( 0, sum_abs_td_error, batch_size )
        rand_list = np.sort( rand_list )
        #print( "rand_list : ", rand_list )

        # 作成した乱数で、TD誤差の絶対値を串刺しにして、サンプリング対象のインデックスを求める
        indexes = []
        idx = 0
        tmp_sum_abs_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_abs_td_error < rand_num:
                tmp_sum_abs_td_error += ( abs(self._memory_td_error[idx]) + self._td_error_epsilon )
                idx += 1

            # 微小値を計算に使用した関係でindexがメモリの長さを超えた場合の補正
            if idx >= len(self._memory_td_error):
                idx = len(self._memory_td_error) - 1

            indexes.append(idx)

        return indexes


    def get_mini_batch( self, batch_size, episode ):
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
        # Prioritized　Experience Replay に基づき、ミニバッチ処理用のデータセットを生成する。
        #----------------------------------------------------------------------
        # エピソードのはじめは、TD誤差がないので、通常の ExperienceRelpay を使用する。
        if( episode < self._change_episode ):
            # ミニバッチサイズ以上ならば、学習用データを pop する
            batch  = self.pop( batch_size )
            #print( "batch  :", batch  )

        # ある程度エピソードが経過してから、Prioritized　Experience Replay を使用する。
        else:
            indexes = self.get_prioritized_indexes( batch_size )
            batch  = [ self._memory[n] for n in indexes ]

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
            [ torch.from_numpy(b.state).type(torch.FloatTensor).unsqueeze(0) for b in batch ], 
            dim=0
        ).to(self._device)

        action_batch = torch.cat(
            [ torch.LongTensor([[b.action]]) for b in batch ]   # [[x]] で shape = [batch_size, 1] にしておく
        ).to(self._device)

        reward_batch = torch.cat(
            [ torch.FloatTensor([b.reward]) for b in batch ]
        ).to(self._device)

        next_state_batch = torch.cat(
            [ torch.from_numpy(b.next_state).type(torch.FloatTensor).unsqueeze(0) for b in batch ], 
            dim=0
        ).to(self._device)

        done_batch = torch.cat(
            [ torch.FloatTensor([b.done]) for b in batch ]
        ).to(self._device)

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch


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
        #----------------------------------------------------------------------
        # Prioritized　Experience Replay に基づき、ミニバッチ処理用のデータセットを生成する。
        #----------------------------------------------------------------------
        # エピソードのはじめは、TD誤差がないので、通常の ExperienceRelpay を使用する。
        if( episode < self._change_episode ):
            # ミニバッチサイズ以上ならば、学習用データを pop する
            batch  = self.pop( batch_size )
            #print( "batch  :", batch  )

        # ある程度エピソードが経過してから、Prioritized　Experience Replay を使用する。
        else:
            indexes = self.get_prioritized_indexes( batch_size )
            batch  = [ self._memory[n] for n in indexes ]

        # 取り出したデータをミニバッチ学習用に reshape
        # transtions : shape = 1 step 毎の (s,a,s',r) * batch_size / shape = 32 * 4
        # → shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = 4 * 32
        batch = Transition( *zip(*batch ) )
        #print( "batch :", batch )

        # torch.cat() : Tensorをリスト入れてして渡すことで、それらを連結したTensorを返す。連結する軸はdimによって指定。
        #               Tensor の配列(shepe=batch_size)である batch[0]=batch.state → 1つの Tensor である state.batch に変換
        state_batch = torch.cat( batch.state )
        action_batch = torch.cat( batch.action )
        reward_batch = torch.cat( batch.reward )
        next_state_batch = torch.cat( batch.next_state )
        done_batch = torch.cat( batch.done )

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch


    def get_td_error_mini_batch( self ):
        """
        """
        #--------------------------------------------------------------------
        # 全メモリでミニバッチを作成
        #--------------------------------------------------------------------
        batch  = self._memory

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
            [ torch.from_numpy(b.state).type(torch.FloatTensor).unsqueeze(0) for b in batch ], 
            dim=0
        ).to(self._device)

        action_batch = torch.cat(
            [ torch.LongTensor([[b.action]]) for b in batch ]   # [[x]] で shape = [batch_size, 1] にしておく
        ).to(self._device)

        reward_batch = torch.cat(
            [ torch.FloatTensor([b.reward]) for b in batch ]
        ).to(self._device)

        next_state_batch = torch.cat(
            [ torch.from_numpy(b.next_state).type(torch.FloatTensor).unsqueeze(0) for b in batch ], 
            dim=0
        ).to(self._device)

        done_batch = torch.cat(
            [ torch.FloatTensor([b.done]) for b in batch ]
        ).to(self._device)

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    def get_td_error_mini_batch_from_tensor( self ):
        """
        """
        #--------------------------------------------------------------------
        # 全メモリでミニバッチを作成
        #--------------------------------------------------------------------
        batch  = self._memory

        # 取り出したデータをミニバッチ学習用に reshape
        # transtions : shape = 1 step 毎の (s,a,s',r) * batch_size / shape = 32 * 4
        # → shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = 4 * 32
        batch = Transition( *zip(*batch ) )
        #print( "batch :", batch )

        # torch.cat() : Tensorをリスト入れてして渡すことで、それらを連結したTensorを返す。連結する軸はdimによって指定。
        #               Tensor の配列(shepe=batch_size)である batch[0]=batch.state → 1つの Tensor である state.batch に変換
        state_batch = torch.cat( batch.state )
        action_batch = torch.cat( batch.action )
        reward_batch = torch.cat( batch.reward )
        next_state_batch = torch.cat( batch.next_state )
        done_batch = torch.cat( batch.done )

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch


    def update_td_error( self, updated_td_errors ):
        '''TD誤差の更新'''
        self._memory_td_error = updated_td_errors
