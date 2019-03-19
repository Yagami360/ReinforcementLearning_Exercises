# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境

"""
    更新情報
    [19/03/19] : 新規作成
                ・参考：https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    [xx/xx/xx] : 
"""
import numpy as np

# OpenAI Gym
import gym
from gym import spaces
from gym.spaces.box import Box

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

# OpenCV
import cv2
cv2.ocl.setUseOpenCL(False)     # ?


class NoopResetEnv( gym.Wrapper ):
    """
    強化学習環境のリセット直後の特定の開始状態で大きく依存して学習が進むのを防ぐために、
    強化学習環境のリセット直後、数ステップ間は何も学習しないプロセスを実施する。
    ・OpenAI Gym の env をラッピングして実装している。
    ・atari_wrappers.py と同じ内容

    [public]
        noop_max : <int> 何も学習しないステップ数

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
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """
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
        '''5機とも失敗したら、本当にリセット'''
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs
