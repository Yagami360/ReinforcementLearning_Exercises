# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/11] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

# 自作クラス
from Brain import Brain
from Agent import Agent
from AdavantageMemory import AdavantageMemory
from A2CNetwork import A2CNetwork

# PyTorch
import torch
from torch  import nn   # ネットワークの構成関連
from torch import optim
import torch.nn.functional as F


class CartPoleA2CBrain( Brain ):
    """
    倒立振子課題（CartPole）の Brain。
    ・A2C [Advantage Actor Critic] によるアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _gamma : <float> 割引利得の γ 値
        _learning_rate : <float> 学習率

        _network : <A2CNetwork> A2C のネットワーク
        _loss_fn : <torch.> モデルの損失関数
        _optimizer : <torch.optimizer> モデルの最適化アルゴリズム

        _memory : <AdavantageMemory> メモリ
    """
    def __init__(
        self,
        n_states,
        n_actions,
        gamma = 0.9, learning_rate = 0.0001,
        batch_size = 32,
        n_ksteps = 5
    ):
        super().__init__( n_states, n_actions )
        self._gamma = gamma
        self._learning_rate = learning_rate

        self._network = None
        self.model()

        self._loss_fn = None
        self._optimizer = None
        self._memory = AdavantageMemory( n_ksteps, n_states, gamma )
        self._b_loss_init = False
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleA2CBrain" )
        print( self )
        print( str )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "_gamma : \n", self._gamma )
        print( "_learning_rate : \n", self._learning_rate )

        print( "_network :\n", self._network )
        print( "_loss_fn :\n", self._loss_fn )
        print( "_optimizer :\n", self._optimizer )
        print( "_memory :\n", self._memory )

        print( "----------------------------------" )
        return

    def get_loss( self ):
        if( self._b_loss_init == True ):
            return self._loss_fn.data
        else:
            return 0.0
        
        
    def action( self, state ):
        """
        Brain のロジックに従って、現在の状態 s での行動 a を決定する。
        
        [Args]
            state : int
                現在の状態
        """
        # ネットワークを推論モードへ切り替え
        self._network.eval()

        with torch.no_grad():
            actor_output, critic_output = self._network( state )
            #print( "actor_output :\n", actor_output )
            #print( "critic_output :\n", critic_output )

            # softmax で確率を計算 / A(s,a) → π(s,a)
            policy = F.softmax( actor_output, dim = 0 )

            # 確率分布で表現された行動方策のうち、最も確率が高い値のインデックスを取得
            action = policy.multinomial( num_samples = 1 )
            #print( "action :", action )

        return action

        
    def model( self ):
        """
        A2C のネットワーク構成を構築する。
        
        [Args]
        [Returns]
        """
        self._network = A2CNetwork(
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )

        print( "_network :", self._network )
        return

    def loss( self ):
        """
        モデルの損失関数を設定する。
        [Args]
        [Returns]
            self._loss_fn : <> モデルの損失関数
        """
        #self._memory.print()

        #-------------------------------------------------------
        # loss 値を計算するために必要な各種値を計算
        # k ステップ間のデータで計算
        #-------------------------------------------------------
        # [0:-1] で最後の行を除外し、[n_kstep+1, 4] → [n_kstep, 4] に reshape したものを渡す
        actor_output, critic_output = self._network( self._memory.observations[0:-1] )
        #print( "self._memory.observations[0:-1] :", self._memory.observations[0:-1] )
        #print( "actor_output :", actor_output )
        #print( "critic_output :", critic_output )

        # π = softmax(A)
        policy = F.softmax( actor_output, dim = 1 )
        #print( "policy :", policy )

        # π = softmax( A(s,a) )
        # log(π)
        log_policy = F.log_softmax( actor_output, dim = 1 )  
        #print( "log_policy :", log_policy )

        # 
        action_log_policy = log_policy.gather( 1, self._memory.actions )
        #print( "action_log_policy :", action_log_policy )

        # L_entropy = Σ_a π*log(π)
        loss_entropy = -( policy * log_policy ).sum(-1).mean()
        #print( "loss_entropy :", loss_entropy )

        # アドバンテージ関数を、割引利得＋割引状態価値関数で近似する。
        advantage = self._memory.total_rewards[0:-1] - critic_output
        #print( "memory / total_reward[0:-1]", self._memory.total_rewards[0:-1] )
        #print( "advantage :", advantage )

        #-------------------------------------------------------
        # 損失関数の計算
        #-------------------------------------------------------
        # アクター側の損失関数 : Loss_actor = E[log(π)*A] - Loss_entorpy
        loss_actor = - ( action_log_policy * advantage.detach() ).mean() - 0.01 * loss_entropy
        #print( "loss_actor :", loss_actor )

        # クリティック側の損失関数 : Loss_critic = {Q(s,a)-V(s)}^2
        loss_critic = advantage.pow(2).mean()
        #print( "loss_critic :", loss_critic )

        # 全損失関数
        self._loss_fn = loss_actor + 0.5 * loss_critic
        #print( "loss_fn :", self._loss_fn )
        
        # loss 値の初回計算済フラグ
        self._b_loss_init = True

        return self._loss_fn


    def optimizer( self ):
        """
        モデルの最適化アルゴリズムを設定する。
        [Args]
        [Returns]
            self._optimizer : <torch.optimizer> モデルの最適化アルゴリズム            
        """
        # 最適化アルゴリズムとして、Adam を採用
        self._optimizer = optim.Adam( 
            params = self._network.parameters(), 
            lr = self._learning_rate 
        )

        return self._optimizer


    def fit( self ):
        """
        モデルを学習し、
        [Args]
        [Returns]
        """
        # 損失関数を計算する
        self.loss()

        # モデルを学習モードに切り替える。
        self._network.train()

        # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
        self._optimizer.zero_grad()

        # 誤差逆伝搬
        self._loss_fn.backward()

        # 一気に重みパラメータ θ が更新されないように、勾配は最大 0.5 までにする。
        nn.utils.clip_grad_norm_( self._network.parameters(), 0.5 )

        # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
        self._optimizer.step()

        #print( "loss :", self._loss_fn.data )

        return


    def update( self, state, action, reward, done_mask ):
        """
        """
        #
        self._memory.insert( state, action, reward, done_mask )
        
        #
        self.fit()

        #------------------------------------------------------
        # Meory の k-step 間での割引利得＋状態価値関数を再計算
        #------------------------------------------------------
        # ネットワークを推論モードへ切り替え
        self._network.eval()

        with torch.no_grad():
            actor_output, critic_output = self._network( state )

        # detach() で deep copy したものを、メモリに保管
        v_function = critic_output.detach()
        self._memory.update( v_function )

        #self._memory.print("")

        return



