# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/02/12] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# 自作クラス
from Brain import Brain
from Agent import Agent

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch  import nn   # ネットワークの構成関連
from torch import optim
import torch.nn.functional as F
import torchvision      # 画像処理関連


class CartPoleDQNBrain( Brain ):
    """
    倒立振子課題（CartPole）の Brain。
    ・DQN によるアルゴリズム
    
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _n_dizitzed : <int> 各状態を離散化する分割数
        _epsilon : <float> ε-greedy 法の ε 値
        _gamma : <float> 割引利得の γ 値
        _learning_rate : <float> 学習率

        _q_function : <float,float> 行動状態関数 Q(s,a) / 行を状態 s, 列を行動 a とする表形式表現
        _expected_q_function : <float,float> 推定行動状態関数 Q(s,a) / 行を状態 s, 列を行動 a とする表形式表現

        _model : <torch.nn.Sequential> モデルのオブジェクト
        _loss_fn : <torch.> モデルの損失関数
        _optimizer : <torch.optimizer> モデルの最適化アルゴリズム

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        n_states,
        n_actions,
        epsilon = 0.5, gamma = 0.9, learning_rate = 0.0001 ,
        n_dizitzed = 6
    ):
        super().__init__( n_states, n_actions )
        self._n_dizitzed = n_dizitzed
        self._epsilon = epsilon
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._model = None
        self._loss_fn = None
        self._optimizer = None

        # Qテーブルの作成（行数＝分割数^状態数=6^4=）
        self._q_function = np.random.uniform(
            low = 0, high =1,
            size = ( self._n_dizitzed ** self._n_states, self._n_actions )
        )
        self._expected_q_function = np.random.uniform(
            low = 0, high =1,
            size = ( self._n_dizitzed ** self._n_states, self._n_actions )
        )

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleDQNBrain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "_n_dizitzed : \n", self._n_dizitzed )
        print( "_epsilon : \n", self._epsilon )
        print( "_gamma : \n", self._gamma )
        print( "_learning_rate : \n", self._learning_rate )

        print( "_q_function : \n", self._q_function )
        print( "_expected_q_function : \n", self._expected_q_function )

        print( "_model :\n", self._model )
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
        # Sequential モデルでネットワーク構成
        #------------------------------------------------
        self._model = nn.Sequential()
        self._model.add_module( name = "fc1", module = nn.Linear( in_features = self._n_states, out_features = 32 ) )
        self._model.add_module( "relu1", nn.ReLU() )
        self._model.add_module( "fc2", nn.Linear( 32, 32 ) )
        self._model.add_module( "relu2", nn.ReLU() )
        self._model.add_module( "fc3", nn.Linear( 32, self._n_actions ) )

        print( "model :", self._model )


    def loss( self ):
        """
        モデルの損失関数を設定する。
        [Args]
        [Returns]
            self._loss_fn : <> モデルの損失関数
        """
        # smooth L1 関数（＝Huber 関数）
        """
        self._loss_fn = F.smooth_l1_loss( 
            input = self._q_function,           # 行動価値関数 Q(s,a;θ) / ミニバッチデータ
            target = self._expected_q_function  # 推定行動価値関数 Q(s,a;θ)
        )
        """

        print( "loss_fn :", self._loss_fn )
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
            params = self._model.parameters(), 
            lr = self._learning_rate 
        )

        return self._optimizer


    def predict( self ):
        """
        教師信号となる行動価値関数を求める

        [Args]
        [Returns]
        """
        # ネットワークを推論モードへ切り替え（PyTorch特有の処理）
        self._model.eval()

        # 構築したDQNのネットワークが出力する Q(s,a) を求める。

    def fit( self ):
        """
        """
        return



        return

    def decay_learning_rate( self ):
        """
        学習率を減衰させる。
        """
        self._learning_rate = self._learning_rate / 2.0
        return

    def decay_epsilon( self, episode ):
        """
        ε-greedy 法の ε 値を減衰させる。
        """
        #self._epsilon = self._epsilon / 2.0
        self._epsilon = 0.5 * ( 1 / (episode + 1) )
        return

    def digitize_state( self, observations ):
        """
        観測したエージェントの observations を離散化する。
        """
        def bins( clip_min, clip_max, num ):
            return np.linspace( clip_min, clip_max, num + 1 )[1:-1]

        cart_pos, cart_v, pole_angle, pole_v = observations

        digitized = [
            np.digitize( cart_pos, bins = bins(-2.4, 2.4, self._n_dizitzed) ),
            np.digitize( cart_v, bins = bins(-3.0, 3.0, self._n_dizitzed) ),
            np.digitize( pole_angle, bins = bins(-0.5, 0.5, self._n_dizitzed) ),
            np.digitize( pole_v, bins = bins(-2.0, 2.0, self._n_dizitzed) )
        ]
        return sum( [x * (self._n_dizitzed**i) for i, x in enumerate(digitized)] )


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
            # Q の最大化する行動を選択
            action = np.argmax( self._q_function[state, :] )
        else:
            # ε の確率でランダムな行動を選択
            action = np.random.choice( self._n_actions )

        return action


    def update_q_function( self, state, action, next_state, reword ):
        """
        Q 関数の値を更新する。

        [Args]
            state : <int> 現在の状態 s のインデックス
            action : <int> 現在の行動 a
            next_state : <int> 次の状態 s'
            reword : <float> 報酬
        
        [Returns]

        """
        q_funtion_next = max( self._q_function[next_state][:] )
        self._q_function[ state, action ] += self._learning_rate * ( reword + self._gamma * q_funtion_next - self._q_function[ state, action ] )

        return self._q_function

