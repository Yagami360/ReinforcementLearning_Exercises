# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/18] : 新規作成
    [xx/xx/xx] : 
"""
import numpy as np

# OpenAI Gym
import gym

# PyTorch
import torch

# 自作クラス
from Agent import Agent


class BreakoutAgent( Agent ):
    """
    OpenAIGym の Breakout のエージェント

    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <torch.device> 実行デバイス

        _env : OpenAI Gym の ENV
        _losses : list<float> 損失関数の値のリスト（長さはエピソード長）

    """
    def __init__( 
        self,
        device,
        env,
        brain = None, 
        gamma = 0.9,
        n_stack_frames = 4
    ):
        super().__init__( brain, gamma, 0 )
        self._device = device
        self._env = env        
        self._total_reward = torch.FloatTensor( [0.0] )
        self._loss_historys = []

        # shape = [mini_batch, n_stack_frames(=n_channels), height, width]
        obs_shape = self._env.observation_space.shape  # (1, 84, 84)
        self._observations = torch.zeros( 1, n_stack_frames, obs_shape[1], obs_shape[2] )
        self._observations_next = torch.zeros( 1, n_stack_frames, obs_shape[1], obs_shape[2] )
        return

    def print( self, str ):
        print( "----------------------------------" )
        print( "BreakoutAgent" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_env :", self._env )
        print( "_brain : \n", self._brain )
        print( "_observations : \n", self._observations )
        print( "_observations_next : \n", self._observations_next )
        print( "_total_reward : \n", self._total_reward )
        print( "_gamma : \n", self._gamma )
        print( "_done : \n", self._done )
        print( "_s_a_historys : \n", self._s_a_historys )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def get_loss_historys( self ):
        return self._loss_historys

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        obs_new = self._env.reset()   # shape = [1,84,84]
        obs_new = torch.from_numpy( obs_new ).float().to(self._device)    # numpy → Tensor に型変換
        obs_new = torch.unsqueeze( obs_new, dim = 0 ).to(self._device)  # ミニバッチ用の次元を追加
        self._observations = obs_new

        self._total_reward = torch.FloatTensor( [0.0] )
        self._done = False
        return

    def agent_step( self, episode, time_step ):
        """
        エージェント [Agent] の次の状態を決定する。
        ・Academy から各時間ステップ度にコールされるコールバック関数

        [Args]
            episode : 現在のエピソード数
            time_step : 現在の時間ステップ

        [Returns]
            done : bool
                   エピソードの完了フラグ
        """
        # 既にエピソードが完了状態なら、そのまま return して、全エージェントの完了を待つ
        if( self._done == True):
            return self._done

        #-------------------------------------------------------------------
        # ε-greedy 法の ε 値を減衰させる。
        #-------------------------------------------------------------------
        #self._brain.decay_epsilon()
        self._brain.decay_epsilon_episode( episode )

        #-------------------------------------------------------------------
        # 行動 a_t を求める
        #-------------------------------------------------------------------
        action = self._brain.action( self._observations, time_step )
        #print( "action :", action )

        #-------------------------------------------------------------------
        # 行動を実行する。
        #-------------------------------------------------------------------
        observations_next, reward, env_done, info = self._env.step( action.item() )

        # numpy → PyTorch 用の型に変換
        observations_next = torch.from_numpy(observations_next).float().to(self._device)

        # ミニバッチ学習用の次元を追加
        observations_next = torch.unsqueeze( observations_next, dim = 0 ).to(self._device)

        #print( "reward :", reward )
        #print( "env_done :", env_done )
        #print( "info :", info )

        #------------------------------------------------------------------
        # 行動の実行により、次の時間での状態 s_{t+1} 報酬 r_{t+1} を求める。
        #------------------------------------------------------------------
        self.add_reward( reward, time_step )
        reward = torch.FloatTensor( [reward] ).to(self._device)

        # env_done :  ⇒ True
        if( env_done == True ):
            # 次の状態は存在しない（＝終端状態）ので、None に設定する
            observations_next = None

        else:
            pass

        #----------------------------------------
        # 価値関数の更新
        #----------------------------------------
        self._brain.update_q_function( self._observations, action, observations_next, reward )

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


    def agent_on_done( self, episode, time_step ):
        """
        Academy のエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
        ・Academy からコールされるコールバック関数

        [Args]
            episode : <int> 現在のエピソード数
        """
        print( "エピソード = {0} / 最終時間ステップ数 = {1}".format( episode, time_step )  )

        # 利得の履歴に追加
        self._reward_historys.append( self._total_reward )

        # 損失関数の履歴に追加
        print( "loss = %0.6f" % self._brain.get_loss() )
        self._loss_historys.append( self._brain.get_loss() )

        # 一定間隔で、Target Network と Main Network を同期する
        if( (episode % 2) == 0 ):
            self._brain.update_target_q_function()

        return
