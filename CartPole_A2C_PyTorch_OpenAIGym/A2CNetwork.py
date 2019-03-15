# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/11] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

class A2CNetwork( nn.Module ):
    """
    A2C のネットワーク構成
    ・PyTorch の nn.Module を継承して実装
    ・簡単のため CNN ではなく、MLP でネットワーク構成

    [public]
        fc1 : [nn.Linear] 入力層
        fc2 : [nn.Linear] 隠れ層
        actor : [nn.Linear] アクター側の出力層
        critic : [nn.Linear] クリティック側の出力層
    """
    def __init__( self, n_states, n_hiddens, n_actions ):
        """
        [Args]
            n_states : 状態数 |S| / 入力ノード数に対応する。
            n_actions : 状態数 |A| / 出力ノード数に対応する。
        """
        super( A2CNetwork, self ).__init__()
        self.fc1 = nn.Linear( n_states, n_hiddens )
        self.fc2 = nn.Linear( n_hiddens, n_hiddens )
        self.actor = nn.Linear( n_hiddens, n_actions )
        self.critic = nn.Linear( n_hiddens, 1 )
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
