# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/04] : 新規作成
    [xx/xx/xx] : 
"""


class Agent( object ):
    """
    強化学習におけるエージェントをモデル化したクラス。
    ・実際の Agent クラスの実装は、このクラスを継承し、オーバーライドするを想定している。

    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _brain : Brain
                エージェントの Brain への参照

        _observations : list<動的な型>
            エージェントが観測できる状態

        _reword : float
            報酬

        _done : bool

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( self, brain = None ):
        self._brain = brain
        self._observations = []
        self._reword = 0.0
        self._done = False
        self.agent_reset()

        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Agent" )
        print( self )
        print( str )

        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_reword : \n", self._reword )
        print( "_done : \n", self._done )
        print( "----------------------------------" )
        return

    def set_brain( self, brain ):
        """
        エージェントの Brain を設定する。
        """
        self._brain = brain
        return

    def add_vector_obs( self, observation ):
        """
        エージェントが観測できる状態を追加する。
        """
        self._observations.append( observation )
        return

    def Done( self ):
        """
        エピソードを完了にする。
        """
        self._done = True
        return

    def IsDone( self ):
        """
        Academy がエピソードを完了したかの取得
        """
        return self._done

    def set_reword( self, reword ):
        """
        報酬をセットする
        """
        self._reword = reword
        return

    def add_reword( self, reword ):
        """
        報酬を加算する
        """
        self._reword += reword
        return

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._observations = []
        self._reword = 0.0
        self._done = False
        return


    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        return


    def agent_step( self, step ):
        """
        エージェント [Agent] の次の状態を決定する。
        ・Academy からコールされるコールバック関数
        [Args]
            step : 学習環境のシミュレーションステップ

        [Returns]
            done : bool
                   シミュレーションの完了フラグ
        """
        done = False
        return done
    
    def agent_action( self, step ):
        """
        現在の状態に基づき、エージェントの実際のアクションを記述する。
        ・Academy からステップ毎にコールされるコールバック関数
        [Args]

        """
        return

    def agent_on_done( self ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数
        """
        return

