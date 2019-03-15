# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/01/25] : 新規作成
    [xx/xx/xx] : 
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
        super().__init__( brain, gamma, 0 )
        self._env = env
        
        self._observations = []
        self._total_reward = torch.FloatTensor( [0.0] )
        self._loss_historys = []
        #self._n_succeeded_episode = 0

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
        print( "_s_a_historys : \n", self._s_a_historys )
        print( "_reward_historys : \n", self._reward_historys )
        print( "----------------------------------" )
        return

    def get_num_states( self ):
        """
        エージェントの状態数を取得する
        """
        num_states = self._env.observation_space.shape[0]
        return num_states

    def get_num_actions( self ):
        """
        エージェントの状態数を取得する
        """
        num_actions = self._env.action_space.n
        return num_actions

    def get_loss_historys( self ):
        return self._loss_historys

    def agent_reset( self ):
        """
        エージェントの再初期化処理
        """
        self._observations = self._env.reset()
        self._observations = torch.from_numpy( self._observations ).float()
        self._total_reward = torch.FloatTensor( [0.0] )
        self._done = False
        self._brain.memory.observations[0].copy_( self._observations )
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
        if( self._done == True ):
            return self._done

        #-------------------------------------------------------------------
        # 離散化した現在の状態 s_t を元に、行動 a_t を求める
        #-------------------------------------------------------------------
        action = self._brain.action( self._observations )
        #print( "self._observations :", self._observations )
        #print( "action.item() :", action.item() )
        
        #-------------------------------------------------------------------
        # 行動を実行し、次の状態を得る。
        #-------------------------------------------------------------------
        observations_next, _, env_done, _ = self._env.step( action.item() )

        # numpy →  Tensor に変換
        observations_next = torch.from_numpy( observations_next ).float()
        #print( "env_done :", env_done )
        #print( "info :", info )

        #------------------------------------------------------------------
        # 行動の実行により、次の時間での報酬 r_{t+1} を求める。
        #------------------------------------------------------------------
        reward = torch.FloatTensor( [0.0] )
        # env_done : ステップ数が最大数経過 OR 一定角度以上傾くと ⇒ True
        if( env_done == True ):
            # 時間ステップの最大回数に近づいたら
            if time_step < self._max_time_step - self._n_kstep:
                # 途中でコケたら、報酬－１
                reward = torch.FloatTensor( [-1.0] )
                self.add_reward( reward, time_step )
                #self._n_succeeded_episode = 0
            else:
                # 立ったまま終了時は、報酬＋１
                reward = torch.FloatTensor( [1.0] )
                self.add_reward( reward, time_step )
                #self._n_succeeded_episode += 1
        else:
            # 途中報酬
            pass
            #self.set_reward( reward )
            #reward = torch.FloatTensor( [1.0] )
            #self.add_reward( reward, time_step )

        #----------------------------------------
        # エピソードが完了したかのマスク定数
        #----------------------------------------
        if( env_done == True ):
            done_mask =  torch.FloatTensor( [0.0] )
        else:
            done_mask =  torch.FloatTensor( [1.0] )

        # 完了時は observation を 0 にする
        observations_next *= done_mask
        #print( "observations_next :", observations_next )

        #---------------------------------------------
        # メモリに値を挿入
        #---------------------------------------------
        self._brain.memory.insert( observations_next, action, reward, done_mask )
        #self._brain.memory.print()

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


    def agent_on_kstep_done( self, episode, time_step ):
        """
        """
        #----------------------------------------
        # Brain の更新
        #----------------------------------------
        self._brain.update()

        return


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
        
        return
