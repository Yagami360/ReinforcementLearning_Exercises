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
        _observations : <Tensor> shape = n_states
        _reward : <Tensor>

    [protected] 変数名の前にアンダースコア _ を付ける        
        _n_ksteps : [int] 先読みするステップ数 k
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

        self.observations = torch.zeros( n_ksteps + 1, n_states )
        self.rewards = torch.zeros( n_ksteps, 1 )
        self.actions = torch.zeros( n_ksteps, 1 ).long()
        self.done_masks = torch.ones( n_ksteps + 1, 1 )
        self.total_rewards = torch.zeros( n_ksteps + 1, 1 )
        self._index = 0

        self._gamma = gamma
        return


    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "CartPoleA2CBrain" )
        print( self )
        print( str )
        print( "_index :\n", self._index )
        print( "observations :\n", self.observations )
        print( "rewards :\n", self.rewards )
        print( "actions :\n", self.actions )
        print( "done_masks :\n", self.done_masks )
        print( "total_rewards :\n", self.total_rewards )
        print( "----------------------------------" )
        return

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
        self.observations[self._index + 1].copy_( observations )   # 次の状態 s_{t+1} なので、index+1
        self.rewards[self._index].copy_( reward )   # 
        self.actions[self._index].copy_( action )
        self.done_masks[self._index + 1].copy_( done_mask )

        # index = 0 → 1 → ... → _n_ksteps → 0 → 1,...
        self._index = ( self._index + 1 ) % self._n_ksteps
        return


    def update( self, v_function ):
        """
        割引報酬和＋割引状態価値関数を再計算する。
        """
        self.total_rewards[-1] = v_function
        
        #self.print()
        # 逆順ループ： n_kstep - 1 → ... → 2 → 1 → 0
        for step in reversed( range(self._n_ksteps) ):
            #print( "step :", step )
            self.total_rewards[step] = \
                self.total_rewards[step + 1] * self._gamma * self.done_masks[step + 1] + self.rewards[step]
            
        # 割引報酬和を更新後は、0 番目の要素にコピー
        self.observations[0].copy_( self.observations[-1] )
        self.done_masks[0].copy_( self.done_masks[-1] )

        return

    