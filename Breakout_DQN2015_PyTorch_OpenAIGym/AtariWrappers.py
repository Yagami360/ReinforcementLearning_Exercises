# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/19] : 新規作成
                ・参考：https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    [xx/xx/xx] : 
"""
import numpy as np
from collections import deque

# OpenAI Gym
import gym
from gym import spaces
from gym.spaces.box import Box

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

# OpenCV
import cv2
#cv2.ocl.setUseOpenCL(False)     # ?


class NoopResetEnv( gym.Wrapper ):
    """
    強化学習環境のリセット直後の特定の開始状態で大きく依存して学習が進むのを防ぐために、
    強化学習環境のリセット直後、数ステップ間は何も学習しないプロセスを実施する。
    ・OpenAI Gym の env をラッピングして実装している。
    ・atari_wrappers.py と同じ内容

    [public]
        env : OpenAIGym の ENV
        noop_max : <int> 何も学習しないステップ数
        override_num_noops : 
        noop_action : <int> 何もしない行動の値

    [protected] 変数名の前にアンダースコア _ を付ける

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """
        reset() メソッドをオーバーライド
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        """
        step() メソッドをオーバーライト
        """
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Breakout は５機のライフがあるので、５回失敗でゲーム終了となるが、
    この残機では学習が面倒なので、１回失敗でゲーム終了に設定する。
    但し、１回失敗毎に完全にリセットすると、初期状態ばかり学習してしまい、過学習してしまいやすいので、
    １回失敗でのリセットでは、崩したブロックの状態はそのままにしておいて、
    ５回失敗でのリセットでは、崩したブロックの状態もリセットする完全なリセットとする。
    ・OpenAI Gym の env をラッピングして実装している。
    ・atari_wrappers.py と同じ内容？

    [public]
        env : OpenAIGym の ENV
        lives : <int> 残機
        was_real_done : <bool> 完全なリセットの意味での終了フラグ

    [protected] 変数名の前にアンダースコア _ を付ける

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """
        step() メソッドをオーバーライト
        """
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        reset() メソッドをオーバーライト
        ・5機とも失敗したら、本当にリセット
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Breakout は 60FPS で動作するが、この速さで動かすと早すぎるので、
    ４フレーム単位で行動を判断させ、４フレーム連続で同じ行動を行うようにする。
    これにより、60FPS → 15 FPS となる。
    但し、atari のゲームには、奇数フレームと偶数フレームで現れる画像が異なるゲームがあるために、
    画面上のチラツキを抑える意味で、最後の3、4フレームの最大値をとった画像を observation として採用する。
    ・OpenAI Gym の env をラッピングして実装している。
    ・atari_wrappers.py とほぼ同じ内容

    [public]
        env : OpenAIGym の ENV

    [protected] 変数名の前にアンダースコア _ を付ける
        _obs_buffer :
        _skip : <int> フレームのスキップ数

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """
        step() メソッドをオーバーライト
        Repeat action, sum reward, and max over last observations.
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """
        reset() メソッドをオーバーライト
        """
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    報酬のクリッピング
    """
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    画像サイズをNatureのDQN論文と同じ84x84のグレースケールに reshape する。

    [public]
        env : OpenAIGym の ENV
        observation_space : <spaces.Box> obsevation の shape / 継承先の gym.ObservationWrapper からの変数
        width : <int> 入力画像の幅のリサイズ後の値
        height : <int> 入力画像の高さのリサイズ後の値

    [protected] 変数名の前にアンダースコア _ を付ける

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        """
        observation の Getter をオーバーライド
        """
        # 入力画像をグレースケールに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # (width, height) に reshape
        frame = cv2.resize(
            frame, (self.width, self.height),
            interpolation = cv2.INTER_AREA
        )

        return frame    # [width,height] で return

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.stack(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class WrapFrameStack(gym.Wrapper):
    def __init__(self, env, n_stack_frames = 4 ):
        """
        obsevation を４フレーム分重ねる。
        モデルに一度に入力する画像データのフレーム数
        """
        gym.Wrapper.__init__(self, env)
        self.n_stack_frames = n_stack_frames
        self.frames = deque( [], maxlen = n_stack_frames )  # deque 構造で４フレーム分だけ確保
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box( 
            low = 0, high = 255, 
            shape = ( obs_shape[0], obs_shape[1], obs_shape[2] * n_stack_frames )
        )
        return


    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stack_frames):
            self.frames.append( obs )

        # list 部分を numpy 化
        #observation = np.array( self.frames )
        #observation = list( self.frames )

        observation = LazyFrames( list(self.frames) )
        observation = np.array( observation )

        return observation

    def step(self, action):
        """
        step() メソッドをオーバーライト
        """
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)

        # list 部分を numpy 化
        #observation = np.array( self.frames )
        #observation = list( self.frames )
        observation = LazyFrames( list(self.frames) )
        observation = np.array( observation )

        return observation, reward, done, info


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class WrapMiniBatch(gym.ObservationWrapper):
    """
    obsevation を ミニバッチ学習用のインデックス順に reshape する。
    [width, height, n_channels(=n_skip_frames)] → [n_channels(=n_skip_frames), width, height] 
    """
    def __init__(self, env=None):
        super(WrapMiniBatch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape = [obs_shape[2], obs_shape[0], obs_shape[1] ],
            dtype=self.observation_space.dtype
        )
        return

    def observation(self, observation):
        """
        observation の Getter をオーバーライド
        """
        # [width, height, n_channels(=n_skip_frames)] → [n_channels(=n_skip_frames), width, height] に reshape
        #observation[0] = observation[0].transpose(0, 1)
        #observation[1] = observation[0].transpose(0, 1)
        #observation[2] = observation[0].transpose(0, 1)
        #observation[3] = observation[0].transpose(0, 1)
        observation = observation.transpose(2, 0, 1)
        return observation


def make_env( device, env_id, seed = 8, n_noop_max = 30, n_skip_frame = 4, n_stack_frames = 4 ):
    """
    上記 Wrapper による強化学習環境の生成メソッド

    [Args]
        device : <torch.device> 実行デバイス
        env_id : 
    """
    # OpenAI Gym の強化学習環境生成
    env = gym.make(env_id)

    # env をラッピングしていくことで、独自の設定を適用する。
    env = NoopResetEnv( env, noop_max = n_noop_max )
    env = MaxAndSkipEnv( env, skip = n_skip_frame )
    env.seed(seed)                  # 乱数シードの設定
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = ClipRewardEnv(env)
    env = WrapFrameStack(env, n_stack_frames )
    #env = WrapMiniBatch(env)

    return env
