# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/02/14] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

class DuelingNetworkMLP3( nn.Module ):
    """
    Dueling Network のネットワーク構成
    PyTorch の nn.Module を継承して実装

    [public]
        fc1 : [nn.Linear] 入力層
        fc2 : [nn.Linear] 隠れ層
        fc3_adv : [nn.Linear] アドバンテージ関数のネットワーク
        fc3_vfunc : [nn.Linear] 状態価値関数のネットワーク
    """
    def __init__( self, n_states, n_hiddens, n_actions ):
        """
        [Args]
            n_states : 状態数 |S| / 入力ノード数に対応する。
            n_actions : 状態数 |A| / 出力ノード数に対応する。
        """
        super( DuelingNetworkMLP3, self ).__init__()
        self.fc1 = nn.Linear( in_features = n_states, out_features = n_hiddens )
        self.fc2 = nn.Linear( in_features = n_hiddens, out_features = n_hiddens )
        self.fc3_adv = nn.Linear( in_features = n_hiddens, out_features = n_actions )
        self.fc3_vfunc = nn.Linear( in_features = n_hiddens, out_features = 1 )
        return

    def forward( self, x ):
        """
        ネットワークの順方向での更新処理
        """
        h1 = F.relu( self.fc1(x) )
        h2 = F.relu( self.fc2(h1) )

        # アドバンテージ関数のネットワークからの出力（この出力は Reluしない）
        adv = self.fc3_adv( h2 )
        #print( "adv :", adv )

        # 推定状態価値関数のネットワークからの出力（この出力は Reluしない）
        # アドバンテージ関数の値 adv とか加算処理するために、[batch_size*1]→[batch_size*2] に展開
        # adv.size(1) : 行動数=2
        v_func = self.fc3_vfunc(h2).expand( -1, adv.size(1) )
        #print( "v_func :",v_func )

        # 行動価値関数を output
        # Q(s,a;θ,α,β)=V(s;θ,β)+{ A(s,a)+1/|行動数|*Σ_a' A(s,a') }
        # adv.mean( 1, keepdim=True ) : 列方向（行動）で平均
        # expand( -1, adv.size(1) ) : [batch_size*1]→[batch_size*2] に展開
        output = v_func + adv - adv.mean( 1, keepdim=True ).expand( -1, adv.size(1) )
        #print( "output :", output )

        return output
