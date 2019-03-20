# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/18] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

class Flatten( nn.Module) :
    '''コンボリューション層の出力画像を1次元に変換する層を定義'''

    def forward(self, x):
        return x.view(x.size(0), -1)


class QNetworkCNN( nn.Module ):
    """
    DQN のネットワーク構成
    PyTorch の nn.Module を継承して

    [public]

    """
    def __init__( self, device, in_channles, n_actions ):
        """
        [Args]
            _device : <torch.device> 実行デバイス

            in_channles : チャンネル数（＝入力画像データの枚数）
            n_actions : 状態数 |A| / 出力ノード数に対応する。
        """
        super( QNetworkCNN, self ).__init__()
        self._device = device

        self.layer = nn.Sequential(
            nn.Conv2d( in_channels = in_channles, out_channels = 32, kernel_size = 8, stride = 4 ),
            nn.ReLU(),
            nn.Conv2d( in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2 ),
            nn.ReLU(),
            nn.Conv2d( in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1 ),
            nn.ReLU(),
            Flatten(),
            #nn.Linear( in_features = 64*7*7, out_features = 512 ),
            nn.Linear( in_features = 64*7*7, out_features = n_actions ),
            nn.ReLU()
        )

        return

    def forward( self, x ):
        """
        ネットワークの順方向での更新処理
        """
        # 画像のピクセル値0-255を0-1に正規化する
        x = x / 255.0

        output = self.layer(x)

        return output
