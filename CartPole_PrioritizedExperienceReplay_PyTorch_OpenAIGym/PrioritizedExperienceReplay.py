# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/07] : 新規作成
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


class PrioritizedExperienceReplay( object ):
    """
    Prioritized　Experience Replay による学習用のミニバッチデータの生成を行うクラス
    ・サンプリング優先度は、TD誤差で判断する。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける        
        _capacity : [int] メモリの最大値
        _memory : [list] (s,a,s',a',r) のリスト（学習用データ）
        _index : [int] 現在のメモリのインデックス

        _memory_td_error : [list] サンプリング優先度を表すTD誤差のメモリ
        _index_td_error : [int] 現在のメモリのインデックス
        _td_error_epsilon : [float] サンプリング優先度を表すTD誤差のバイアス値

        _change_episode : Experience Replay → PrioritizedExperienceReplay に切り替えるエピソード数

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        capacity = 10000,
        td_error_epsilon = 0.0001,
        change_episode = 30
    ):
        self._capacity = capacity
        self._memory = []
        self._index = 0

        self._memory_td_error = []
        self._index_td_error = 0
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
        print( "_index : ", self._index )
        print( "_memory_td_error :\n", self._memory_td_error )
        print( "_index_td_error : ", self._index_td_error )
        print( "_td_error_epsilon : ", self._td_error_epsilon )
        print( "_change_episode : ", self._change_episode )
        return

    def get_memory( self ):
        return self._memory

    def set_td_error_memory( self, td_errors ):
        self._memory_td_error = td_errors

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

    def push_td_error( self, td_error ):
        """
        サンプリング優先度を表すTD誤差をメモリに格納する。

        [Args]
            td_error : [float] TD誤差
        """
        # 現在のメモリサイズが上限値以下なら、新たに容量を確保する。
        if( len(self._memory_td_error) < self._capacity ):
            self._memory_td_error.append( None )

        # メモリに値を格納
        self._memory_td_error[ self._index_td_error ] = td_error

        # 現在のインデックスをづらす
        self._index_td_error = ( self._index_td_error + 1 ) % self._capacity

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
        """
        #----------------------------------------------------------------------
        # Prioritized　Experience Replay に基づき、ミニバッチ処理用のデータセットを生成する。
        #----------------------------------------------------------------------
        # メモリサイズがまだミニバッチサイズより小さい場合は、処理を行わない
        if( len(self._memory) < batch_size ):
            return None, None, None, None, None

        # エピソードのはじめは、TD誤差がないので、通常の ExperienceRelpay を使用する。
        if( episode < self._change_episode ):
            # ミニバッチサイズ以上ならば、学習用データを pop する
            transitions = self.pop( batch_size )
            #print( "transitions :", transitions )

        # ある程度エピソードが経過してから、Prioritized　Experience Replay を使用する。
        else:
            indexes = self.get_prioritized_indexes( batch_size )
            transitions = [ self._memory[n] for n in indexes ]

        # 取り出したデータをミニバッチ学習用に reshape
        # transtions : shape = 1 step 毎の (s,a,s',r) * batch_size / shape = 32 * 4
        # → shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = 4 * 32
        batch = Transition( *zip(*transitions) )
        #print( "batch :", batch )

        #
        state_batch = torch.cat( batch.state )
        action_batch = torch.cat( batch.action )
        reward_batch = torch.cat( batch.reward )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        return batch, state_batch, action_batch, reward_batch, non_final_next_states


    def get_td_error_mini_batch( self ):
        """
        """
        #--------------------------------------------------------------------
        # 全メモリでミニバッチを作成
        #--------------------------------------------------------------------
        transitions = self._memory
        batch = Transition( *zip(*transitions) )

        # 取り出したデータをミニバッチ学習用に reshape
        # transtions : shape = 1 step 毎の (s,a,s',r) * batch_size / shape = 32 * 4
        # → shape = (s * batch_size, a * batch_size, s' * batch_size, r * batch_size) / shape = 4 * 32
        batch = Transition( *zip(*transitions) )

        #
        state_batch = torch.cat( batch.state )
        action_batch = torch.cat( batch.action )
        reward_batch = torch.cat( batch.reward )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        return batch, state_batch, action_batch, reward_batch, non_final_next_states


    def update_td_error( self, updated_td_errors ):
        '''TD誤差の更新'''
        self._memory_td_error = updated_td_errors
