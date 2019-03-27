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

class QNetworkMLP4( nn.Module ):
    """
    DQN のネットワーク構成
    PyTorch の nn.Module を継承して

    [public]
    """
    def __init__( self, device, n_states, n_hiddens, n_actions ):
        """
        [Args]
            n_states : 状態数 |S| / 入力ノード数に対応する。
            n_actions : 状態数 |A| / 出力ノード数に対応する。
        """
        super( QNetworkMLP4, self ).__init__()
        self._device = device

        # ４層
        self.layer = nn.Sequential(
            nn.Linear( in_features = n_states, out_features = n_hiddens ),
            nn.ReLU(),
            nn.Linear( in_features = n_hiddens, out_features = n_hiddens ),
            nn.ReLU(),
            nn.Linear( in_features = n_hiddens, out_features = n_hiddens ),
            nn.ReLU(),
            nn.Linear( in_features = n_hiddens, out_features = n_actions )
        ).to(self._device)

        return

    def forward( self, x ):
        """
        ネットワークの順方向での更新処理
        """
        output = self.layer(x)
        return output
