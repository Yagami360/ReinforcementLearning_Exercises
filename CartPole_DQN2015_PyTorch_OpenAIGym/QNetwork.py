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

class QNetwork( nn.Module ):
    """
    DQN のネットワーク構成
    PyTorch の nn.Module を継承して

    [public]
        fc1 : [nn.Linear] 入力層
        fc2 : [nn.Linear] 隠れ層
        fc3 : 出力層
    """
    def __init__( self, n_inputs, n_hiddens, n_outputs ):
        super( QNetwork, self ).__init__()
        self.fc1 = nn.Linear( in_features = n_inputs, out_features = n_hiddens )
        self.fc2 = nn.Linear( in_features = n_hiddens, out_features = n_hiddens )
        self.fc3 = nn.Linear( in_features = n_hiddens, out_features = n_outputs )
        return

    def forward( self, x ):
        """
        ネットワークの順方向での更新処理
        """
        h1 = F.relu( self.fc1(x) )
        h2 = F.relu( self.fc2(h1) )
        output = self.fc3(h2)
        return output
