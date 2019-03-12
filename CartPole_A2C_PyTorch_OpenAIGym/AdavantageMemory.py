# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/11] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

# PyTorch
import torch


class AdavantageMemory( object ):
    """
    Advantage 学習用のメモリ
    ・kステップ数でのメモリ領域を循環参照することで、kステップ先読みでの学習に対応

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける        
        _n_processes : [int] 同時に実行する環境数
        _n_ksteps : [int] 先読みするステップ数 k

        _observations : <Tensor> shape = n_states
        _reward : <Tensor>

        _index : <int> 現在のメモリ先のインデックス

        _gamma : <float> 割引利得の γ 値
    """
    def __init__( 
        self, 
        n_ksteps, n_states,
        gamma = 0.0001
    ):
        #self._n_processes = n_processes
        self._n_ksteps = n_ksteps

        self._observations = torch.zeros( n_ksteps + 1, n_states )
        self._rewards = torch.zeros( n_ksteps, 1 )
        self._actions = torch.zeros( n_ksteps, 1 ).long()
        self._done_masks = torch.ones( n_ksteps + 1, 1 )
        self._total_rewards = torch.zeros( n_ksteps + 1, 1 )
        self._index = 0

        self._gamma = gamma
        return


    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "CartPoleA2CBrain" )
        print( self )
        print( str )
        print( "_index :\n", self._index )
        print( "_observations :\n", self._observations )
        print( "_rewards :\n", self._rewards )
        print( "_actions :\n", self._actions )
        print( "_done_masks :\n", self._done_masks )
        print( "_total_rewards :\n", self._total_rewards )
        print( "----------------------------------" )
        return

    def get_total_reward( self ):
        return self._total_rewards[-1]

    def insert( self, observations, action, reward, done_mask ):
        """
        メモリの末端にデータを挿入する
        [Args]
            observations : <Tensor> 状態 / shape = [n_states]
            reward : <Tensor> 報酬
            done_mask : 
        """
        # .copy_() : Deep Copy
        # _で終わるメソッドは呼び出し元の変数の値を変化させる
        self._observations[self._index + 1].copy_( observations )   # 次の状態 s_{t+1} なので、index+1
        self._rewards[self._index].copy_( reward )   # 
        self._actions[self._index].copy_( action )
        self._done_masks[self._index + 1].copy_( done_mask )

        # index = 0 → 1 → ... → _n_ksteps → 0 → 1,...
        self._index = ( self._index + 1 ) % self._n_ksteps
        return


    def update( self, v_function ):
        """
        割引報酬和＋割引状態価値関数を再計算する。
        """
        self._total_rewards[-1] = v_function
        
        #self.print()
        # 逆順ループ： n_kstep - 1 → ... → 2 → 1 → 0
        for step in reversed( range(self._n_ksteps) ):
            #print( "step :", step )
            self._total_rewards[step] = \
                self._total_rewards[step + 1] * self._gamma * self._done_masks[step + 1] + self._rewards[step]
            
        # 割引報酬和を更新後は、0 番目の要素にコピー
        self._observations[0].copy_( self._observations[-1] )
        self._done_masks[0].copy_( self._done_masks[-1] )

        return

    