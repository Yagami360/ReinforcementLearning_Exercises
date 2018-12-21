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

        _action : サブクラスにて動的に型を決定する
                  エージェント取りうるアクション（上移動、下移動など）

        _observations : list<動的な型>
                エージェントが観測できる状態

        _policy : 動的な型
                行動方策 π。確率値（0~1）

        _brain_parameters : 動的な型
                行動方策 π を決定するためのパラメーター
                
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self ):
        self._agent = None
        self._action = None
        self._observations = None
        self._policy = None
        self._brain_parameters = None
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Brain" )
        print( self )
        print( str )
        print( "_agent : \n", self._agent )
        print( "_action : \n", self._action )
        print( "_observations : \n", self._observations )
        print( "_policy : \n", self._policy )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "----------------------------------" )
        return

    def reset_brain( self ):
        """
        """
        self._policy = None
        self._brain_parameters = None
        return

    def get_policy( self ):
        return self._policy

        
    def set_agent( self, agent ):
        self._agent = agent
        return

    def action( self ):
        """
        Action を取得
        """
        return self._action

    def decision_policy( self ):
        """
        行動方針を決定する
        """
        # エージェントの状態を取得
        self._observations = self._agent.collect_observations()

        # 行動の方策のためのパラメーターを更新

        # 行動の方策のためのパラメーターを元に、行動方策を決定する。

        return self._policy