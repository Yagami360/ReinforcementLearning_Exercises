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

class QNetworkMLP3( nn.Module ):
    """
    DQN のネットワーク構成
    PyTorch の nn.Module を継承して

    [public]
        fc1 : [nn.Linear] 入力層
        fc2 : [nn.Linear] 隠れ層
        fc3 : 出力層
    """
    def __init__( self, n_states, n_hiddens, n_actions ):
        """
        [Args]
            n_states : 状態数 |S| / 入力ノード数に対応する。
            n_actions : 状態数 |A| / 出力ノード数に対応する。
        """
        super( QNetworkMLP3, self ).__init__()
        self.fc1 = nn.Linear( in_features = n_states, out_features = n_hiddens )
        self.fc2 = nn.Linear( in_features = n_hiddens, out_features = n_hiddens )
        self.fc3 = nn.Linear( in_features = n_hiddens, out_features = n_actions )
        return

    def forward( self, x ):
        """
        ネットワークの順方向での更新処理
        """
        h1 = F.relu( self.fc1(x) )
        h2 = F.relu( self.fc2(h1) )
        output = self.fc3(h2)
        return output
