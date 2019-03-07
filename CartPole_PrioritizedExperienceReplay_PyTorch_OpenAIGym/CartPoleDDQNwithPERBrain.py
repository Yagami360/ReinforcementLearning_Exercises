# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/07] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np
import random

# 自作クラス
from Brain import Brain
from Agent import Agent
from PrioritizedExperienceReplay import PrioritizedExperienceReplay
from QNetworkMLP3 import QNetworkMLP3

# PyTorch
import torch
from torch  import nn   # ネットワークの構成関連
from torch import optim
import torch.nn.functional as F


class CartPoleDDQNwithPERBrain( Brain ):
    """
    倒立振子課題（CartPole）の Brain。
    ・DDQN & Prioritized Experience Replay によるアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _epsilon : <float> ε-greedy 法の ε 値
        _gamma : <float> 割引利得の γ 値
        _learning_rate : <float> 学習率

        _q_function : <> 教師信号である古いパラメーター θ- で固定化された行動状態関数 Q(s,a,θ-)
        _expected_q_function : <> 推定行動状態関数 Q(s,a,θ)
        _memory : <PrioritizedExperienceReplay> PrioritizedExperienceReplay に基づく学習用のデータセット

        _main_network : <QNetwork> DDQNのメインネットワーク
        _target_network : <QNetwork> DDQNのターゲットネットワーク

        _loss_fn : <torch.> モデルの損失関数
        _optimizer : <torch.optimizer> モデルの最適化アルゴリズム

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        n_states,
        n_actions,
        epsilon = 0.5, gamma = 0.9, learning_rate = 0.0001,
        batch_size = 32,
        memory_capacity = 10000,
        td_error_epsilon = 0.0001,
        change_episode = 30
    ):
        super().__init__( n_states, n_actions )
        self._epsilon = epsilon
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._main_network = None
        self._target_network = None
        self.model()

        self._loss_fn = None
        self._optimizer = None

        self._q_function = None
        self._expected_q_function = None

        self._memory = PrioritizedExperienceReplay( 
            capacity = memory_capacity, 
            td_error_epsilon = td_error_epsilon,
            change_episode = change_episode
        )

        self._b_loss_init = False

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleDDQNBrain" )
        print( self )
        print( str )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "_epsilon : \n", self._epsilon )
        print( "_gamma : \n", self._gamma )
        print( "_learning_rate : \n", self._learning_rate )
        print( "_batch_size : \n", self._batch_size )

        print( "_q_function : \n", self._q_function )
        print( "_expected_q_function : \n", self._expected_q_function )
        print( "_memory :", self._memory )

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
        self._main_network = QNetworkMLP3(
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )

        self._target_network = QNetworkMLP3(
            n_states = self._n_states, 
            n_hiddens = 32,
            n_actions = self._n_actions
        )

        print( "main network :", self._main_network )
        print( "target network :", self._target_network )
        return

    def loss( self ):
        """
        モデルの損失関数を設定する。
        [Args]
        [Returns]
            self._loss_fn : <> モデルの損失関数
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


    def predict( self, batch, state_batch, action_batch, reward_batch, non_final_next_states ):
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
        outputs = self._main_network( state_batch )
        #print( "outputs :", outputs )

        # outputs から実際にエージェントが選択した action を取り出す
        # gather(...) : 
        # dim = 1 : 列方向
        # index = action_batch : エージェントが実際に選択した行動は action_batch 
        self._q_function = outputs.gather( 1, action_batch )
        #print( "_q_function :", self._q_function )

        #--------------------------------------------------------------------
        # CartPole が done ではなく、next_state が存在するインデックス用のマスク
        #--------------------------------------------------------------------
        non_final_mask = torch.ByteTensor(
            tuple( map(lambda s: s is not None,batch.next_state) )
        )
        #print( "non_final_mask :", non_final_mask )

        #--------------------------------------------------------------------
        # 次の状態 s_{t+1} でQ値を最大化する行動 a_m を求める
        # a_m = arg max_{a∈A} Q_main(s_{t+1},a)
        #--------------------------------------------------------------------
        a_m = torch.zeros( self._batch_size ).type(torch.LongTensor)
        #print( "a_m :", a_m )   # tensor([0, 0, 0,...,0) / shape = [batch_size]

        # a_m は、メインネットワークから求める
        # tensor.detach() : 新しい Tensor を返す。このとき、Variable の誤差逆伝搬による値の更新が止まる。
        # 教師信号は固定された値である必要があるので、detach() で値が変更させないようにする。
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self._main_network( non_final_next_states ).detach().max(1)[1]
        #print( "a_m :", a_m )   # shape = [batch_size]

        # 次の状態があるものだけにフィルターし、batch_size → batch_size * 1 へ reshape
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        #print( "a_m_non_final_next_states :", a_m_non_final_next_states )   # shape = [batch_size*1]

        #--------------------------------------------------------------------
        # 次の状態を求める
        #--------------------------------------------------------------------
        # 全部 0 で初期化
        next_state_values = torch.zeros( self._batch_size )

        # Main Network ではなく Target Network からの出力
        next_outputs = self._target_network( non_final_next_states )
        #print( "next_outputs :", next_outputs )

        # tensor.detach() : 新しい Tensor を返す。このとき、Variable の誤差逆伝搬による値の更新が止まる。
        # 教師信号は固定された値である必要があるので、detach() で値が変更させないようにする。
        # squeeze() で [batch_size]→[batch_size*1] に reshape
        #next_state_values[non_final_mask] = next_outputs.max(1)[0].detach()
        next_state_values[non_final_mask] = next_outputs.gather(1, a_m_non_final_next_states).detach().squeeze()
        #print( "next_state_values :", next_state_values )

        #--------------------------------------------------------------------
        # ネットワークの出力となる推定行動価値関数を求める
        #--------------------------------------------------------------------
        self._expected_q_function = reward_batch + self._gamma * next_state_values

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

    def decay_epsilon( self, episode ):
        """
        ε-greedy 法の ε 値を減衰させる。
        """
        #self._epsilon = self._epsilon / 2.0
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

                # .view(1,1) : [torch.LongTensor of size 1] → size 1×1 に reshape
                action = max_index.view(1,1)
                #print( "action :", action )

        else:
            # ε の確率でランダムな行動を選択
            #action = np.random.choice( self._n_actions )
            action = torch.LongTensor(
                [ [random.randrange(self._n_actions)] ]
            )

        return action


    def update_q_function( self, state, action, next_state, reward, episode ):
        """
        Q 関数の値を更新する。

        [Args]
            state : <int> 現在の状態 s のインデックス
            action : <int> 現在の行動 a
            next_state : <int> 次の状態 s'
            reword : <float> 報酬
            episode : 現在のエピソード数
        
        [Returns]

        """
        #-----------------------------------------
        # 経験に基づく学習用データを追加
        #-----------------------------------------
        self._memory.push( state = state, action = action, next_state = next_state, reward = reward )

        # TD誤差メモリにTD誤差を追加
        # 本当はTD誤差を格納するが、0をいれておく
        self._memory.push_td_error(0)

        # 学習用データがミニバッチサイズ以下ならば、以降の処理は行わない
        if( len(self._memory) < self._batch_size ):
            return

        #-----------------------------------------        
        # ミニバッチデータを取得する
        #-----------------------------------------
        batch, state_batch, action_batch, reward_batch, non_final_next_states = self._memory.get_mini_batch( self._batch_size, episode )

        #-----------------------------------------
        # 教師信号となる推定行動価値関数を求める 
        #-----------------------------------------
        self.predict( batch, state_batch, action_batch, reward_batch, non_final_next_states )

        #-----------------------------------------
        # ネットワークを学習し、パラメーターを更新する。
        #-----------------------------------------
        self.fit()

        return self._q_function


    def update_target_q_function( self ):
        """
        Target Network を Main Network と同期する。
        """
        # load_state_dict() : モデルを読み込み
        self._target_network.load_state_dict(
            state_dict = self._main_network.state_dict()    # Main Network のモデルを読み込む
        )

        return

    def update_td_error_memory( self ):
        """
        TD誤差メモリに格納されているTD誤差を更新する
        """
        #--------------------------------------------------------------------
        # ネットワークを推論モードへ切り替え（PyTorch特有の処理）
        #--------------------------------------------------------------------
        self._main_network.eval()
        self._target_network.eval()

        #-----------------------------------------        
        # ミニバッチデータを取得する
        #-----------------------------------------
        batch, state_batch, action_batch, reward_batch, non_final_next_states = self._memory.get_td_error_mini_batch()

        #--------------------------------------------------------------------
        # 構築したDQNのネットワークが出力する Q(s,a) を求める。
        # 学習用データをモデルに流し込む
        # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
        #--------------------------------------------------------------------
        # outputs / shape = batch_size * _n_actions
        outputs = self._main_network( state_batch )

        # outputs から実際にエージェントが選択した action を取り出す
        # gather(...) : 
        # dim = 1 : 列方向
        # index = action_batch : エージェントが実際に選択した行動は action_batch 
        self._q_function = outputs.gather( 1, action_batch )

        #--------------------------------------------------------------------
        # CartPole が done ではなく、next_state が存在するインデックス用のマスク
        #--------------------------------------------------------------------
        non_final_mask = torch.ByteTensor(
            tuple( map(lambda s: s is not None,batch.next_state) )
        )

        #--------------------------------------------------------------------
        # 次の状態 s_{t+1} でQ値を最大化する行動 a_m を求める
        # a_m = arg max_{a∈A} Q_main(s_{t+1},a)
        #--------------------------------------------------------------------
        a_m = torch.zeros( len( self._memory ) ).type(torch.LongTensor)
        #print( "a_m :", a_m )   # tensor([0, 0, 0,...,0) / shape = [batch_size]

        # a_m は、メインネットワークから求める
        # tensor.detach() : 新しい Tensor を返す。このとき、Variable の誤差逆伝搬による値の更新が止まる。
        # 教師信号は固定された値である必要があるので、detach() で値が変更させないようにする。
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self._main_network( non_final_next_states ).detach().max(1)[1]
        #print( "a_m :", a_m )   # shape = [batch_size]

        # 次の状態があるものだけにフィルターし、batch_size → batch_size * 1 へ reshape
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        #print( "a_m_non_final_next_states :", a_m_non_final_next_states )   # shape = [batch_size*1]

        #--------------------------------------------------------------------
        # 次の状態を求める
        #--------------------------------------------------------------------
        # 全部 0 で初期化
        next_state_values = torch.zeros( len( self._memory ) )

        # Main Network ではなく Target Network からの出力
        next_outputs = self._target_network( non_final_next_states )
        #print( "next_outputs :", next_outputs )

        # tensor.detach() : 新しい Tensor を返す。このとき、Variable の誤差逆伝搬による値の更新が止まる。
        # 教師信号は固定された値である必要があるので、detach() で値が変更させないようにする。
        # squeeze() で [batch_size]→[batch_size*1] に reshape
        #next_state_values[non_final_mask] = next_outputs.max(1)[0].detach()
        next_state_values[non_final_mask] = next_outputs.gather(1, a_m_non_final_next_states).detach().squeeze()
        #print( "next_state_values :", next_state_values )

        #--------------------------------------------------------------------
        # TD誤差を求める
        #--------------------------------------------------------------------
        # state_action_valuesはsize[minibatch×1]なので、squeezeしてsize[minibatch]へ
        td_errors = (reward_batch + self._gamma * next_state_values) - self._q_function.squeeze()
        
        # TD誤差メモリを更新、Tensorをdetach()で取り出し、NumPyにしてから、Pythonのリストまで変換
        td_errors_list = td_errors.detach().numpy().tolist()
        self._memory.set_td_error_memory( td_errors_list )

        return
