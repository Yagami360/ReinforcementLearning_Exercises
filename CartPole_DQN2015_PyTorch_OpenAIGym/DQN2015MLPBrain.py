# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/02/12] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

# 自作クラス
from Brain import Brain
from Agent import Agent
from ExperienceReplay import ExperienceReplay
from QNetworkMLP3 import QNetworkMLP3
from QNetworkMLP4 import QNetworkMLP4

# PyTorch
import torch
from torch  import nn   # ネットワークの構成関連
from torch import optim
import torch.nn.functional as F


class DQN2015MLPBrain( Brain ):
    """
    DQN (2015年バージョン ; MLPベース）の Brain。
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける

        _epsilon : <float> ε-greedy 法の ε 値
        _gamma : <float> 割引利得の γ 値
        _learning_rate : <float> 学習率

        _q_function : <Tensor> 教師信号である古いパラメーター θ- で固定化された行動状態関数 Q(s,a,θ-)
        _expected_q_function : <Tesnor> 推定行動状態関数 Q(s,a,θ)
        _memory : <ExperienceRelay> ExperienceRelayに基づく学習用のデータセット

        _main_network : <QNetwork> モデルのネットワーク
        _target_network : <QNetwork> DQNのターゲットネットワーク
        _loss_fn : <torch.> モデルの損失関数
        _optimizer : <torch.optimizer> モデルの最適化アルゴリズム

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device,
        n_states, n_actions,
        epsilon_init = 1.0, epsilon_final = 0.1, n_epsilon_step = 1000000,
        epsilon = 0.5, gamma = 0.9, learning_rate = 0.0001,
        batch_size = 32,
        memory_capacity = 10000,
        n_frec_target_update = 50
    ):
        super().__init__( n_states, n_actions )
        self._device = device
        self._epsilon = epsilon_init
        self._epsilon_init = epsilon_init
        self._epsilon_final = epsilon_final
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._n_frec_target_update = n_frec_target_update

        self._main_network = None
        self._target_network = None
        self.model()
        self._loss_fn = None
        self._optimizer = None

        self._q_function = None
        self._expected_q_function = None
        self._memory = ExperienceReplay( device = device, capacity = memory_capacity )

        self._b_loss_init = False
        self._epsilon_step = ( epsilon_init - epsilon_final ) / n_epsilon_step

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "DQN2015MLPBrain" )
        print( self )
        print( str )
        print( "_device : ", self._device )
        print( "_n_states : ", self._n_states )
        print( "_n_actions : ", self._n_actions )
        print( "_epsilon : ", self._epsilon )
        print( "_epsilon_init : ", self._epsilon_init )
        print( "_epsilon_final : ", self._epsilon_final )
        print( "_epsilon_step : ", self._epsilon_step )
        print( "_gamma : ", self._gamma )
        print( "_learning_rate : ", self._learning_rate )
        print( "_batch_size : ", self._batch_size )
        print( "_n_frec_target_update : ", self._n_frec_target_update )
        #print( "_q_function : \n", self._q_function )
        #print( "_expected_q_function : \n", self._expected_q_function )
        #print( "_memory :", self._memory )

        print( "_main_network :\n", self._main_network )
        print( "_target_network :\n", self._target_network )
        print( "_loss_fn :\n", self._loss_fn )
        print( "_optimizer :\n", self._optimizer )

        print( "----------------------------------" )
        return

    def get_q_function( self ):
        """
        Q 関数の値を取得する。
        """
        return self._q_function

    def get_epsilon( self ):
        return self._epsilon


    def model( self ):
        """
        DQN のネットワーク構成を構築する。
        
        [Args]
        [Returns]
        """
        self._main_network = QNetworkMLP3(
            device = self._device,
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )
        self._target_network = QNetworkMLP3(
            device = self._device,
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )
        """
        self._main_network = QNetworkMLP4(
            device = self._device,
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )
        self._target_network = QNetworkMLP4(
            device = self._device,
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )
        """
        return


    def loss( self ):
        """
        モデルの損失関数を設定する。
        [Args]
        [Returns]
            self._loss_fn : モデルの損失関数
        """
        # smooth L1 関数（＝Huber 関数）
        self._loss_fn = F.smooth_l1_loss( 
            input = self._q_function,                        # 行動価値関数 Q(s,a;θ) / shape = n_batch
            target = self._expected_q_function.unsqueeze(1)  # 推定行動価値関数 Q(s,a;θ)
        )

        #print( "loss_fn :", self._loss_fn )

        # loss 値の初回計算フラグ
        self._b_loss_init = True

        return self._loss_fn

    def get_loss( self ):
        if( self._b_loss_init == True ):
            return self._loss_fn.data
        else:
            return 0.0


    def optimizer( self ):
        """
        モデルの最適化アルゴリズムを設定する。
        [Args]
        [Returns]
            self._optimizer : <torch.optimizer> モデルの最適化アルゴリズム            
        """
        # 最適化アルゴリズムとして、Adam を採用
        self._optimizer = optim.Adam( 
            params = self._main_network.parameters(), 
            lr = self._learning_rate 
        )

        return self._optimizer

    def predict( self, state_batch, action_batch, next_state_batch, reward_batch, done_batch ):
        """
        教師信号となる行動価値関数を求める

        [Args]
        [Returns]
        """
        #--------------------------------------------------------------------
        # ネットワークを推論モードへ切り替え（PyTorch特有の処理）
        #--------------------------------------------------------------------
        self._main_network.eval()
        self._target_network.eval()

        #--------------------------------------------------------------------
        # 構築したDQNのネットワークが出力する Q(s,a) を求める。
        # 学習用データをモデルに流し込む
        # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
        #--------------------------------------------------------------------
        # outputs / shape = batch_size * _n_actions
        #outputs = self._main_network( state_batch ).to(self._device)
        outputs = self._main_network( state_batch )
        #print( "outputs :", outputs )

        # outputs から実際にエージェントが選択した action を取り出す
        # gather(...) : 
        # dim = 1 : 列方向
        # index = action_batch : エージェントが実際に選択した行動は action_batch 
        #self._q_function = outputs.gather( 1, action_batch ).to(self._device)
        self._q_function = outputs.gather( 1, action_batch )
        #print( "_q_function :", self._q_function )

        #--------------------------------------------------------------------
        # 次の状態を求める
        #--------------------------------------------------------------------
        # Main Network ではなく Target Network からの出力
        #next_outputs = self._target_network( next_state_batch ).to(self._device)
        next_outputs = self._target_network( next_state_batch )
        #print( "next_outputs :", next_outputs )

        # detach() : ネットワークの出力の値を取り出す。Variable の誤差逆伝搬による値の更新が止まる？
        # 教師信号は固定された値である必要があるので、detach() で値が変更させないようにする。
        #next_q_function = next_outputs.max(dim=1)[0].detach().to(self._device)
        next_q_function = next_outputs.max(dim=1)[0].detach()
        #print( "next_q_function :", next_q_function )

        #--------------------------------------------------------------------
        # ネットワークの出力となる推定行動価値関数を求める
        #--------------------------------------------------------------------
        gamma_tsr = torch.FloatTensor( [self._gamma] ).to(self._device)

        # done = 0 ⇒ 価値関数の更新は行われる
        # done = 1 ⇒ 価値関数の更新は行われない
        self._expected_q_function = reward_batch + gamma_tsr * next_q_function * ( 1 - done_batch )

        return



    def fit( self ):
        """
        モデルを学習し、
        [Args]
        [Returns]
        """
        # モデルを学習モードに切り替える。
        self._main_network.train()

        # 損失関数を計算する
        self.loss()

        # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
        self._optimizer.zero_grad()

        # 誤差逆伝搬
        self._loss_fn.backward()

        # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
        self._optimizer.step()

        #print( "loss :", self._loss_fn.data )

        return

    def decay_epsilon( self ):
        """
        ε-greedy 法の ε 値を減衰させる。
        """
        if( self._epsilon > self._epsilon_final and self._epsilon <= self._epsilon_init ):
            self._epsilon -= self._epsilon_step

        return

    def decay_epsilon_episode( self, episode ):
        if( self._epsilon > self._epsilon_final and self._epsilon <= self._epsilon_init ):
            self._epsilon = 0.5 * ( 1 / (episode + 1) )
        return


    def action( self, state ):
        """
        Brain のロジックに従って、現在の状態 s での行動 a を決定する。
        ・ε-グリーディー法に従った行動選択
        [Args]
            state : int
                現在の状態
        """
        # ε-グリーディー法に従った行動選択
        if( self._epsilon <= np.random.uniform(0,1) ):
            #------------------------------
            # Q の最大化する行動を選択
            #------------------------------
            state = torch.from_numpy( state ).type(torch.FloatTensor).to(self._device)      # numpy → Tensor に型変換
            state = torch.unsqueeze( state, dim = 0 ).to(self._device)                      # ミニバッチ用の次元を追加

            # model を推論モードに切り替える（PyTorch特有の処理）
            self._main_network.eval()

            # 微分を行わない処理の範囲を with 構文で囲む
            with torch.no_grad():
                # テストデータをモデルに流し込み、モデルの出力を取得する
                outputs = self._main_network( state )
                #print( "outputs :", outputs )
                #print( "outputs.data :", outputs.data )

                # dim = 1 ⇒ 列方向で最大値をとる
                # Returns : (Tensor, LongTensor)
                _, max_index = torch.max( outputs.data, dim = 1 )
                #print( "max_index :", max_index )

                # tensor → int に変換
                action = max_index.item()
                #print( "action :", action )

        else:
            # ε の確率でランダムな行動を選択
            action = np.random.choice( self._n_actions )

        return action


    def update( 
        self, 
        state, action, next_state, reward, done, 
        episode, time_step, total_time_step
    ):
        """
        Brain の状態を更新する。

        [Args]
            state : <ndarray> 現在の状態 s / shape = [n_channels, width, height]
            action : <int> 現在の行動 a / shape = [1]
            next_state : <ndarray> 次の状態 s' / shape = [n_channels, width, height]
            reword : <float> 報酬
            done : <bool> 完了フラグ
        
        [Returns]

        """
        #-----------------------------------------
        # 経験に基づく学習用データを追加
        #-----------------------------------------
        self._memory.push( state = state, action = action, next_state = next_state, reward = reward, done = done )
         
        # 学習用データがミニバッチサイズ以下ならば、以降の処理は行わない
        if( len(self._memory) < self._batch_size ):
            return

        #-----------------------------------------        
        # ミニバッチデータを取得する
        #-----------------------------------------
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self._memory.get_mini_batch( self._batch_size )

        #-----------------------------------------
        # 教師信号となる推定行動価値関数を求める 
        #-----------------------------------------
        self.predict( state_batch, action_batch, next_state_batch, reward_batch, done_batch )

        #-----------------------------------------
        # ネットワークを学習し、パラメーターを更新する。
        #-----------------------------------------
        self.fit()

        #--------------------------------------------------------
        # 一定間隔で、Target Network と Main Network を同期する
        #--------------------------------------------------------
        self.update_target_q_function( episode, time_step, total_time_step )

        return self._q_function


    def update_target_q_function( self, episode, time_step, total_time_step ):
        """
        Target Network を Main Network と同期する。
        """
        # 一定間隔で同期する。
        #if( (episode % 2) == 0 ):
        if( (total_time_step % self._n_frec_target_update) == 0 ):
            # load_state_dict() : モデルを読み込み
            self._target_network.load_state_dict(
                state_dict = self._main_network.state_dict()    # Main Network のモデルを読み込む
            )

            # Target Network の勾配計算を行わないようにする。別途必要？
            for param in self._target_network.parameters():
                param.requires_grad = False

        return
