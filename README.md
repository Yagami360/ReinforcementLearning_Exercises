# ReinforcementLearning_Exercises
強化学習の練習コード集。<br>
分かりやすいように、各フォルダ毎で完結したコード一式になっています。<br>

強化学習の背景理論は、以下のサイトに記載してあります。<br>

- [【星の本棚】強化学習](http://yagami12.hatenablog.com/entry/2019/02/22/210608)


## ■ 動作環境

- Windows 10
- Geforce GTX 1050 / 2GB
- Python : 3.6
- Anaconda : 5.0.1
- OpenAI Gym : 0.10.9
- PyTorch : 1.0.1

## ■ 項目（フォルダ別）

1. 迷路探索問題
    1. [./MazeSimple_Ramdom](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/MazeSimple_Ramdom)
    1. [./MazeSimple_Sarsa](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/MazeSimple_Sarsa)
    1. [./MazeSimple_Qlearning](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/MazeSimple_Qlearning)
    1. [./MazeSimple_PolicyGD](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/MazeSimple_PolicyGD)
    1. [./MazeSimple_EveryVisitMontecarlo](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/MazeSimple_EveryVisitMontecarlo)
    1. ./MazeSimple_FirstVisitMontecarlo
    1. ./MazeSimple_ActorCritic
1. CartPole（倒立振子課題）
    1. ./CartPole_Sarsa_OpenAIGym
    1. [./CartPole_Qleaning_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_Qlearning_OpenAIGym)
    1. [./CartPole_DQN2013_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_DQN2013_PyTorch_OpenAIGym)
    1. [./CartPole_DQN2015_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_DQN2015_PyTorch_OpenAIGym)
    1. [./CartPole_DDQN_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_DDQN_PyTorch_OpenAIGym)
    1. [./CartPole_PrioritizedExperienceReplay_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_PrioritizedExperienceReplay_PyTorch_OpenAIGym)
    1. [./CartPole_DuelingNetwork_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_DuelingNetwork_PyTorch_OpenAIGym)
    1. [./CartPole_A2C_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/CartPole_A2C_PyTorch_OpenAIGym)
1. FrozenLake
    1. [./FrozenLake_Ramdom_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/FrozenLake_Ramdom_OpenAIGym)
    1. ./FrozenLake_Sarsa_OpenAIGym
    1. [./FrozenLake_Qlearning_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/FrozenLake_Qlearning_OpenAIGym)
    1. ./FrozenLake_EveryVisitMonteCarlo_OpenAIGym
1. Breakout（ブロック崩しゲーム）
    1. [./Breakout_DQN2015_PyTorch_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/Breakout_DQN2015_PyTorch_OpenAIGym)
1. OpenAI Gym
    1. [./Processing_OpenAIGym](https://github.com/Yagami360/ReinforcementLearning_Exercises/tree/master/Processing_OpenAIGym)
1. Unity ML-Agents
    1. ./MazeGame_Unity_ML-Agents

## ■ 参考文献＆サイト

- [つくりながら学ぶ！深層強化学習 PyTorchによる実践プログラミング](https://www.amazon.co.jp/%E3%81%A4%E3%81%8F%E3%82%8A%E3%81%AA%E3%81%8C%E3%82%89%E5%AD%A6%E3%81%B6%EF%BC%81%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-PyTorch%E3%81%AB%E3%82%88%E3%82%8B%E5%AE%9F%E8%B7%B5%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0-%E6%A0%AA%E5%BC%8F%E4%BC%9A%E7%A4%BE%E9%9B%BB%E9%80%9A%E5%9B%BD%E9%9A%9B%E6%83%85%E5%A0%B1%E3%82%B5%E3%83%BC%E3%83%93%E3%82%B9-%E5%B0%8F%E5%B7%9D%E9%9B%84%E5%A4%AA%E9%83%8E-ebook/dp/B07DZVRXFK?SubscriptionId=AKIAJMYP6SDQFK6N4QZA&amp;tag=cloudstudy09-22&amp;linkCode=xm2&amp;camp=2025&amp;creative=165953&amp;creativeASIN=B07DZVRXFK)
    - [Deep-Reinforcement-Learning-Book](https://github.com/Yagami360/Deep-Reinforcement-Learning-Book)<br>

- [機械学習スタートアップシリーズ Pythonで学ぶ強化学習 入門から実践まで (KS情報科学専門書)](https://www.amazon.co.jp/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-Python%E3%81%A7%E5%AD%A6%E3%81%B6%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-%E5%85%A5%E9%96%80%E3%81%8B%E3%82%89%E5%AE%9F%E8%B7%B5%E3%81%BE%E3%81%A7-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E4%B9%85%E4%BF%9D/dp/4065142989/ref=sr_1_1?ie=UTF8&qid=1550659636&sr=8-1&keywords=%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92+python)
    - [baby-steps-of-rl-ja](https://github.com/icoxfog417/baby-steps-of-rl-ja)

- [Unityはじめる機械学習・強化学習 Unity ML-Agents実践ゲームプログラミング](https://www.amazon.co.jp/Unity%E3%81%A7%E3%81%AF%E3%81%98%E3%82%81%E3%82%8B%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%BB%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92-Unity-ML-Agents%E5%AE%9F%E8%B7%B5%E3%82%B2%E3%83%BC%E3%83%A0%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0-%E5%B8%83%E7%95%99%E5%B7%9D-%E8%8B%B1%E4%B8%80/dp/4862464181?SubscriptionId=AKIAJMYP6SDQFK6N4QZA&amp&tag=cloudstudy09-22&amp&linkCode=xm2&amp&camp=2025&amp&creative=165953&amp&creativeASIN=4862464181)<br>

<!--
- [Reinforcement Learning in Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Design.md)<br>
- [ML-Agents（ver0.5）の環境導入方法まとめ（Windows版）](http://enjoy-unity.net/ml-agents/ver0-5_matome/)<br>
-->
