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
        _agent : Agent
            この Brain を持つ Agent への参照
        _actions : サブクラスにて動的に型を決定する
                  エージェント取りうるアクション（上移動、下移動など）のリスト
        _states : サブクラスにて動的に型を決定する
                  エージェント取りうる状態のリスト
        _observations : list<動的な型>
                エージェントが観測できる状態
        _policy : 動的な型
                行動方策 π。確率値（0~1）
                
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        states,
        actions
    ):
        self._agent = None
        self._states = states
        self._actions = actions
        self._observations = []
        self._policy = None
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Brain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_states : \n", self._states )
        print( "_actions : \n", self._actions )
        print( "_observations : \n", self._observations )
        print( "_policy : \n", self._policy )
        print( "----------------------------------" )
        return

    def reset_brain( self ):
        """
        Brain を再初期化する
        """
        self._policy = None
        return

    def get_policy( self ):
        return self._policy

        
    def set_agent( self, agent ):
        self._agent = agent
        return

    def get_actions( self ):
        """
        エージェント取りうるアクション Action のリストを取得
        """
        return self._actions

    def decision_policy( self ):
        """
        行動方針を決定する
        """
        # エージェントの状態を取得
        self._observations = self._agent.collect_observations()

        # 行動の方策のためのパラメーターを更新

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。

        return self._policy
