# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/04] : 新規作成
    [xx/xx/xx] : 
"""


class AgentBase( object ):
    """
    強化学習におけるエージェントをモデル化したクラス。
    ・実際の Agent クラスの実装は、このクラスを継承し、オーバーライドするを想定している。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける

        _state : int
            エージェントの状態 s
        _states_history : list <state>
            エージェントの状態の履歴

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self ):
        """
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        """
        self.agent_reset(  )

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "AgentBase" )
        print( self )
        print( str )
        
        print( "_state : \n", self._state )
        print( "_states_history : \n", self._states_history )
        print( "----------------------------------" )
        return

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._state = 0
        self._states_history = []
        self._states_history.append( self._state )
        
        return

    def agent_step( self ):
        """
        エージェント [Agent] の次の状態を記述する。

        [Args]

        [Returns]
            next_state : Agent の次の状態
        """
        next_state = self._state
        return next_state
    
    def agent_action( self ):
        """
        現在の状態に基づき、エージェントの実際のアクションを記述する。

        [Input]

        """
        return