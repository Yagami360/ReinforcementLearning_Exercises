# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/11] : 新規作成
    [19/04/01] : GPU で実行できるように変更 
"""
import numpy as np

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

class A2CMLPNetwork( nn.Module ):
    """
    A2C のネットワーク構成（MLPベース）
    ・PyTorch の nn.Module を継承して実装
    ・簡単のため CNN ではなく、MLP でネットワーク構成

    [public]
        fc1 : [nn.Linear] 入力層
        fc2 : [nn.Linear] 隠れ層
        actor : [nn.Linear] アクター側の出力層
        critic : [nn.Linear] クリティック側の出力層
    """
    def __init__( self, device, n_states, n_hiddens, n_actions ):
        """
        [Args]
            device : 実行デバイス
            n_states : 状態数 |S| / 入力ノード数に対応する。
            n_actions : 状態数 |A| / 出力ノード数に対応する。
        """
        super( A2CMLPNetwork, self ).__init__()
        self._device = device
        self.fc1 = nn.Linear( n_states, n_hiddens ).to(self._device)
        self.fc2 = nn.Linear( n_hiddens, n_hiddens ).to(self._device)
        self.actor = nn.Linear( n_hiddens, n_actions ).to(self._device)
        self.critic = nn.Linear( n_hiddens, 1 ).to(self._device)
        return


    def forward( self, x ):
        """
        ネットワークの順方向での更新処理
        """
        h1 = F.relu( self.fc1(x) )
        h2 = F.relu( self.fc2(h1) )

        critic_output = self.critic( h2 )        
        actor_output = self.actor( h2 )

        return actor_output, critic_output
