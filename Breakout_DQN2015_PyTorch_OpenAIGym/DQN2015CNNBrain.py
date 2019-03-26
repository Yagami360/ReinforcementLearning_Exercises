# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/18] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

# 自作クラス
from Brain import Brain
from Agent import Agent
from ExperienceReplay import ExperienceReplay
from QNetworkCNN import QNetworkCNN

# PyTorch
import torch
from torch  import nn   # ネットワークの構成関連
from torch import optim
import torch.nn.functional as F


class DQN2015CNNBrain( Brain ):
    """
    DQN (2015年バージョン ; CNNベース）の Brain。
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _device : 実行デバイス

        _epsilon : <float> ε-greedy 法の ε 値
        _gamma : <float> 割引利得の γ 値
        _learning_rate : <float> 学習率

        _q_function : <Tensor> 教師信号である古いパラメーター θ- で固定化された行動状態関数 Q(s,a,θ-)
        _expected_q_function : <Tensor> 推定行動状態関数 Q(s,a,θ)
        _memory : <ExperienceRelay> ExperienceRelayに基づく学習用のデータセット

        _main_network : <QNetwork> DQNのネットワーク
        _target_network : <QNetwork> DQNのターゲットネットワーク

        _loss_fn : <torch.> モデルの損失関数
        _optimizer : <torch.optimizer> モデルの最適化アルゴリズム

        _n_stack_frames : <int> モデルに一度に入力する画像のフレーム数

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device,
        n_states, n_actions,
        epsilon_init = 1.0, epsilon_final = 0.1, n_epsilon_step = 1000000,
        gamma = 0.9, learning_rate = 0.0001,
        batch_size = 32,
        memory_capacity = 10000,
        n_stack_frames = 4,
        n_skip_frames = 4,
        n_frec_target_update = 10000
    ):
        super().__init__( n_states, n_actions )
        self._device = device
        self._epsilon = epsilon_init
        self._epsilon_init = epsilon_init
        self._epsilon_final = epsilon_final
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._n_stack_frames = n_stack_frames
        self._n_skip_frames = n_skip_frames
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
        self._action_repeat = 0

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "AtariDQN2015Brain" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_n_states : ", self._n_states )
        print( "_n_actions : ", self._n_actions )
        print( "_epsilon : ", self._epsilon )
        print( "_epsilon_init : ", self._epsilon_init )
        print( "_epsilon_final : ", self._epsilon_final )
        print( "_epsilon_step : ", self._epsilon_step )
        print( "_gamma : ", self._gamma )
        print( "_learning_rate : ", self._learning_rate )
        print( "_batch_size : ", self._batch_size )
        print( "_n_stack_frames : ", self._n_stack_frames )
        print( "_n_skip_frames : ", self._n_skip_frames )
        print( "_n_frec_target_update : ", self._n_frec_target_update )

        print( "_q_function : \n", self._q_function )
        print( "_expected_q_function : \n", self._expected_q_function )
        print( "len( _memory ) :", len( self._memory ) )

        print( "_main_network :\n", self._main_network )
        print( "_target_network :", self._target_network )
        print( "_loss_fn :\n", self._loss_fn )
        print( "_optimizer :\n", self._optimizer )

        print( "----------------------------------" )
        return

    def get_q_function( self ):
        """
        Q 関数の値を取得する。
        """
        return self._q_function


    def model( self ):
        """
        DQN のネットワーク構成を構築する。
        
        [Args]
        [Returns]
        """
        #------------------------------------------------
        # ネットワーク構成
        #------------------------------------------------        
        self._main_network = QNetworkCNN(
            device = self._device,
            in_channles = self._n_stack_frames,
            n_actions = self._n_actions
        ).to(self._device)

        self._target_network = QNetworkCNN(
            device = self._device,
            in_channles = self._n_stack_frames,
            n_actions = self._n_actions
        ).to(self._device)
        
        #print( "main network :", self._main_network )
        #print( "target network :", self._target_network )
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
        """
        self._optimizer = optim.Adam( 
            params = self._main_network.parameters(), 
            lr = self._learning_rate 
        )
        """
        # 最適化アルゴリズムとして、RMSprop を採用
        self._optimizer = optim.RMSprop(
            params = self._main_network.parameters(), 
            lr = self._learning_rate 
        )

        return self._optimizer


    def predict( self, batch, state_batch, action_batch, reward_batch, non_done_next_states ):
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
        outputs = self._main_network( state_batch ).to(self._device)
        #print( "outputs.size() :", outputs.size() )

        # outputs から実際にエージェントが選択した action を取り出す
        # gather(...) : 
        # dim = 1 : 列方向
        # index = action_batch : エージェントが実際に選択した行動は action_batch 
        self._q_function = outputs.gather( 1, action_batch ).to(self._device)
        #print( "_q_function.size() :", self._q_function.size() )

        #--------------------------------------------------------------------
        # 次の状態を求める
        #--------------------------------------------------------------------
        # 全部 0 で初期化
        next_q_function = torch.zeros( self._batch_size ).to(self._device)

        # エージェントが done ではなく、next_state が存在するインデックス用のマスク
        non_done_mask = torch.ByteTensor(
            tuple( map(lambda s: s is not None,batch.next_state) )
        )

        next_outputs = self._target_network( non_done_next_states ).to(self._device)
        #print( "next_outputs :", next_outputs )

        # detach() : ネットワークの出力の値を取り出す。Variable の誤差逆伝搬による値の更新が止まる？
        # 教師信号は固定された値である必要があるので、detach() で値が変更させないようにする。
        next_q_function[non_done_mask] = next_outputs.max(dim=1)[0].detach().to(self._device)
        #print( "next_q_function :", next_q_function )

        #--------------------------------------------------------------------
        # ネットワークの出力となる推定行動価値関数を求める
        #--------------------------------------------------------------------
        gamma_tsr = torch.FloatTensor( [self._gamma] ).to(self._device)
        self._expected_q_function = reward_batch + gamma_tsr * next_q_function

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


    def action( self, state, time_step ):
        """
        Brain のロジックに従って、現在の状態 s での行動 a を決定する。
        ・ε-グリーディー法に従った行動選択
        [Args]
            state : int
                現在の状態
        """
        # フレームスキップ間は、同じ行動をする。
        if( (time_step % self._n_skip_frames) != 0 ):
            action = self._action_repeat
        else:
            # ε-グリーディー法に従った行動選択
            if( self._epsilon <= np.random.uniform(0,1) ):
                #------------------------------
                # Q の最大化する行動を選択
                #------------------------------
                state = torch.from_numpy( state ).type(torch.FloatTensor).to(self._device)              # numpy → Tensor に型変換
                state = torch.unsqueeze( state, dim = 0 ).to(self._device)      # ミニバッチ用の次元を追加

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

        #print( "action :", action )
        self._action_repeat = action

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
        # メモリに保管する値を Tensor に変換
        # この際に、ミニバッチ用の次元を追加
        #-----------------------------------------
        state = torch.from_numpy( state ).type(torch.FloatTensor).to(self._device)
        state = torch.unsqueeze( state, dim = 0 ).to(self._device)
        action = torch.LongTensor( [[action]] ).to(self._device)                # [[x]] で shape = [1,1] にしておき、ミニバッチ用の次元を用意
        reward = torch.FloatTensor( [[reward]] ).to(self._device)
        next_state = torch.from_numpy( next_state ).type(torch.FloatTensor).to(self._device)
        next_state = torch.unsqueeze( next_state, dim = 0 ).to(self._device)
        if( done == True ):
            next_state = None

        done = torch.FloatTensor( [[done]] ).to(self._device)


        #-----------------------------------------
        # 経験に基づく学習用データを追加
        #-----------------------------------------
        self._memory.push( state = state, action = action, next_state = next_state, reward = reward, done = done )

        # 学習用データがミニバッチサイズ以下ならば、以降の処理は行わない
        if( total_time_step <= self._batch_size ):
        #if( total_time_step <= self._memory._capacity ):
            return
        
        #-----------------------------------------        
        # ミニバッチデータを取得する
        #-----------------------------------------
        batch, state_batch, action_batch, reward_batch, done_batch = self._memory.get_mini_batch( self._batch_size )
        
        # done = True を除いた次状態のデータ
        non_done_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        #print( "state_batch.size() :", state_batch.size() )
        #print( "action_batch.size() :", action_batch.size() )
        #print( "reward_batch.size() :", reward_batch.size() )
        #print( "next_state_batch.size() :", next_state_batch.size() )
        #print( "done_batch.size() :", done_batch.size() )

        # ミニバッチデータの確認
        # shape = [mini_batch, n_channels, width, height ] / torch.Size([32, 4, 84, 84])
        """
        np_state_batch0 = state_batch[0,0,:,:].cpu().numpy()
        np_state_batch1 = state_batch[0,1,:,:].cpu().numpy()
        np_state_batch2 = state_batch[0,2,:,:].cpu().numpy()
        np_state_batch3 = state_batch[0,3,:,:].cpu().numpy()
        import matplotlib.pyplot as plt
        plt.gray()
        plt.imshow( np_state_batch0 )   # imshow() は BGR フォーマット
        plt.show()
        plt.imshow( np_state_batch1 )
        plt.show()
        plt.imshow( np_state_batch2 )
        plt.show()
        plt.imshow( np_state_batch3 )
        plt.show()
        """

        #-----------------------------------------
        # 教師信号となる推定行動価値関数を求める 
        #-----------------------------------------
        self.predict( batch, state_batch, action_batch, reward_batch, non_done_next_states )

        #-----------------------------------------
        # ネットワークを学習し、パラメーターを更新する。
        #-----------------------------------------
        # スキップフレーム間隔で勾配再計算
        if( (total_time_step % self._n_skip_frames) == 0 ):
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