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
        _brain : <Brain> エージェントの Brain への参照
        _observations : list<動的な型> エージェントが観測できる状態
        _total_reward : <float> 割引利得の総和
        _gamma : <float> 収益の割引率
        _done : <bool> エピソードの完了フラグ
        _state : <int> エージェントの現在の状態 s
        _action : <int> エピソードの現在の行動 a
        _reward_historys : list<float> 割引利得の履歴 / shape = [n_episode]

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
        self._total_reward = 0.0
        self._gamma = gamma
        self._done = False
        self._state = state0
        self._action = np.nan
        self._reward_historys = [self._total_reward]
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "Agent" )
        print( self )
        print( str )

        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_total_reward : \n", self._total_reward )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_state : \n", self._state )
        print( "_action : \n", self._action )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def get_reward_historys( self ):
        return self._reward_historys


    def set_brain( self, brain ):
        """
        エージェントの Brain を設定する。
        """
        self._brain = brain
        return

    def done( self ):
        """
        エピソードを完了にする。
        """
        self._done = True
        return

    def is_done( self ):
        """
        Academy がエピソードを完了したかの取得
        """
        return self._done

    def set_total_reword( self, total_reward ):
        """
        報酬をセットする
        """
        self._total_reward = total_reward
        return self._total_reward

    def add_reward( self, reward, time_step ):
        """
        報酬を加算する
        ・割引収益 Rt = Σ_t γ^t r_t+1 になるように報酬を加算する。
        """
        self._total_reward += (self._gamma**time_step) * reward
        return self._total_reward

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._total_reward = 0.0
        self._done = False
        self._state = self._s_a_historys[0][0]
        self._action = self._s_a_historys[0][1]
        self._s_a_historys = [ [ self._state, self._action ] ]
        return

    def agent_step( self, episode, time_step, total_time_step ):
        """
        エージェントの時間ステップ度の処理を記述するコールバック関数
        ・Academy から各時間ステップ度にコールされるコールバック関数

        [Args]
            episode : <int> 現在のエピソード数
            time_step : <int> 現在のエピソードにおける経過時間ステップ数
            total_time_step : <int> 全てのエピソードにおける全経過時間ステップ数

        [Returns]
            done : <bool> エピソードの完了フラグ
        """
        self._done = False
        return self._done
    

    def agent_on_done( self, episode, time_step, total_time_step ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数

        [Args]
            episode : <int> 現在のエピソード数
            time_step : <int> エピソード完了時の経過時間ステップ数数
            total_time_step : <int> 全てのエピソードにおける全経過時間ステップ数
        """
        return

