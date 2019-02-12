# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/05] : 新規作成
    [xx/xx/xx] : 
"""
from Agent import Agent


class Brain( object ):
    """
    エージェントの意思決定ロジック
    ・複数のエージェントが同じ意識決定ロジックを共有出来るように、Brain として class 化する。
    ・移動方法などの Action を設定する。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _agent : <Agent> この Brain を持つ Agent への参照
        _n_states : <int> 状態の要素数
        _n_actions : <int> 行動の要素数
                
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        n_states,
        n_actions
    ):
        self._agent = None
        self._n_states = n_states
        self._n_actions = n_actions
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Brain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_n_states : \n", self._n_states )
        print( "_n_actions : \n", self._n_actions )
        print( "----------------------------------" )
        return

    def reset_brain( self ):
        """
        Brain を再初期化する
        """        
        return

    def set_agent( self, agent ):
        """
        この Brain をもつエージェントを設定する。
        """
        self._agent = agent
        return
