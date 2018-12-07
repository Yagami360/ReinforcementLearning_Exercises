# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/05] : 新規作成
    [xx/xx/xx] : 
"""

class Brain( object ):
    """
    エージェントの意思決定ロジック
    ・複数のエージェントが同じ意識決定ロジックを共有出来るように、Brain として class 化する。
    ・移動方法などの Action を設定する。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _action : サブクラスにて動的に型を決定する
                  エージェント取りうるアクション（上移動、下移動など）

        _observation : list<動的な型>
                エージェントが観測できる状態

        _policy : 動的な型
                行動方策 π。確率値（0~1）

        _brain_parameters : 動的な型
                行動方策 π を決定するためのパラメーター
                
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self ):
        self._action = None
        self._observation = None
        self._policy = None
        self._brain_parameters = None
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Brain" )
        print( self )
        print( str )
        print( "_action : \n", self._action )
        print( "_observation : \n", self._observation )
        print( "_policy : \n", self._policy )
        print( "_brain_parameters : \n", self._brain_parameters )
        print( "----------------------------------" )
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
        return self._policy