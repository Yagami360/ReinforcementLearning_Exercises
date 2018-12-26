# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [18/12/04] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np


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
            収益
        _gamma : float
            収益の割引率

        _done : bool
            エピソードの完了フラグ

        _state : int
            エージェントの現在の状態 s
        _action : int
            エピソードの現在の行動 a
        _s_a_historys : list< [int,int] >
            エピソードの状態と行動の履歴

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__( 
        self, 
        brain = None, 
        gamma = 0.9, 
        state0 = 0
    ):
        self._brain = brain
        self._observations = []
        self._reword = 0.0
        self._gamma = gamma
        self._done = False
        self._state = state0
        self._action = np.nan
        self._s_a_historys = [ [ self._state, self._action ] ]
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Agent" )
        print( self )
        print( str )

        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_reword : \n", self._reword )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_state : \n", self._state )
        print( "_action : \n", self._action )
        print( "_s_a_historys : \n", self._s_a_historys )
        print( "----------------------------------" )
        return

    def collect_observations( self ):
        """
        Agent が観測している State を Brain に提供する。
        ・Brain が、エージェントの状態を取得時にコールバックする。
        """
        self._observations = []
        return self._observations


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
        return self._reword

    def add_reword( self, reword ):
        """
        報酬を加算する
        ・割引収益 Rt = Σ_t γ^t r_t+1 になるように報酬を加算する。
        """
        #
        self._reword += self._gamma * reword        # γ^t * r_t+1
        #self._gamma = self._gamma * self._gamma     # γ^t
        return self._reword


    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._reword = 0.0
        self._done = False
        self._state = self._s_a_historys[0][0]
        self._action = self._s_a_historys[0][1]
        self._s_a_historys = [ [ self._state, self._action ] ]
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
    
    def agent_action( self, episode ):
        """
        現在の状態に基づき、エージェントの実際のアクションを記述する。
        ・Academy から１回のエピソード毎にコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
        [Returns]
            done : bool
                   エピソードの完了フラグ
        """
        done = False
        return done


    def agent_on_done( self ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数
        """
        return

