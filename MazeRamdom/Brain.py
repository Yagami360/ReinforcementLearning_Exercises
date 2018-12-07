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

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self ):
        self._action = None
        self._observation = None
        return

    def action( self ):
        """
        Action を取得
        """
        return self._action

