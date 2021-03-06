# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [19/04/01] : ・Academy クラスとのインターフェイス変更。 
                 ・Tensor 型への変換を行わないように修正
"""
import numpy as np

# OpenAI Gym
import gym

# PyTorch
import torch

# 自作クラス
from Agent import Agent


class CartPoleAgent( Agent ):
    """
    OpenAIGym の CartPole のエージェント

    [protected] 変数名の前にアンダースコア _ を付ける
        _env : OpenAI Gym の ENV
        _losses : list<float> 損失関数の値のリスト（長さはエピソード長）

    """
    def __init__( 
        self,
        env,
        brain = None, 
        gamma = 0.9,
        max_time_step = 200,
        n_kstep = 5
    ):
        super().__init__( brain, gamma )
        self._env = env
        
        self._observations = []
        self._total_reward = 0.0
        self._loss_historys = []

        self._max_time_step = max_time_step
        self._n_kstep = n_kstep
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "CartPoleAgent" )
        print( self )
        print( str )
        print( "_env :", self._env )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_total_reward : \n", self._total_reward )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def get_loss_historys( self ):
        return self._loss_historys

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._observations = self._env.reset()
        self._total_reward = 0.0
        self._done = False
        self._brain.reset_brain( state = self._observations )
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
        # 既にエピソードが完了状態なら、そのまま return して、全エージェントの完了を待つ
        if( self._done == True ):
            return self._done

        #-------------------------------------------------------------------
        # 離散化した現在の状態 s_t を元に、行動 a_t を求める
        #-------------------------------------------------------------------
        action = self._brain.action( self._observations )
        #print( "action :", action )
        
        #-------------------------------------------------------------------
        # 行動を実行し、次の状態を得る。
        #-------------------------------------------------------------------
        observations_next, reward, env_done, info = self._env.step( action )
        #print( "env_done :", env_done )
        #print( "info :", info )

        #------------------------------------------------------------------
        # 行動の実行により、次の時間での報酬 r_{t+1} を求める。
        #------------------------------------------------------------------
        reward = 0.0
        # env_done : ステップ数が最大数経過 OR 一定角度以上傾くと ⇒ True
        if( env_done == True ):
            # 時間ステップの最大回数に近づいたら
            if time_step < self._max_time_step - self._n_kstep:
                # 途中でコケたら、報酬－１
                reward = -1.0
                self.add_reward( reward, time_step )
            else:
                # 立ったまま終了時は、報酬＋１
                reward = 1.0
                self.add_reward( reward, time_step )
        else:
            # 途中報酬
            pass

        #---------------------------------------------
        # メモリに値を挿入
        #---------------------------------------------
        self._brain.update( 
            state = self._observations, action = action, next_state = observations_next, reward = reward, done = env_done, 
            episode = episode, time_step = time_step, total_time_step = total_time_step
        )

        #----------------------------------------
        # 状態の更新
        #----------------------------------------
        self._observations = observations_next

        #----------------------------------------
        # 完了時の処理
        #----------------------------------------
        if( env_done == True ):
            self.done()

        return self._done


    def agent_on_kstep_done( self, episode, time_step, total_time_step ):
        """
        """
        #----------------------------------------
        # Brain の更新
        #----------------------------------------
        self._brain.update_on_kstep_done()

        return


    def agent_on_done( self, episode, time_step, total_time_step ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数
        [Args]
            episode : <int> 現在のエピソード数
            time_step : エピソード完了時の時間ステップ数
            total_time_step : <int> 全てのエピソードにおける全経過時間ステップ数
        """
        # 利得の履歴に追加
        self._reward_historys.append( self._total_reward )

        # 損失関数の履歴に追加
        self._loss_historys.append( self._brain.get_loss() )

        # デバッグ情報出力
        print( "エピソード = {0} / 全時間ステップ数 = {1} / 最終時間ステップ数 = {2}".format( episode, total_time_step, time_step )  )
        print( "total_reward = %0.6f / loss = %0.6f" % (self._total_reward, self._brain.get_loss() ) )
        return
