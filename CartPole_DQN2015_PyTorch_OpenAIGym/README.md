# DQN（2015年NatureバージョンのTargetNetwork使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2015年NatureバージョンのTarget Q-Network使用）によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDQNのネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Deep Q Network（DQN）](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#DeepQNetwork)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9
- PyTorch : 1.0.0

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 500               # エピソード試行回数
NUM_TIME_STEP = 200             # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.0001    # 学習率
BRAIN_BATCH_SIZE = 32           # ミニバッチサイズ
BRAIN_GREEDY_EPSILON = 0.5      # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99             # 利得の割引率
MEMORY_CAPACITY = 10000         # Experience Relay 用の学習用データセットのメモリの最大の長さ
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0に設定|←|
|DQNのネットワーク構成|MLP<br>入力層：状態数<br>隠れ層：32ノード<br>出力層：行動数|MLP<br>入力層：状態数<br>隠れ層１：32ノード<br>隠れ層２：32ノード<br>出力層：行動数|

<!--
転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|
-->

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_dqn2015_reward_episode500](https://user-images.githubusercontent.com/25688193/52898225-a8911980-321e-11e9-9604-fc8fab7fa5da.png)
> DQN2013バージョンより、学習が安定していることがわかる。（Target Q-Network 同期による学習安定化効果？）

- 損失関数のグラフ（実行条件１）<br>
![cartpole-v0_dqn2015_episode500](https://user-images.githubusercontent.com/25688193/52898254-f9a10d80-321e-11e9-9a99-d637f0111f92.png)<br>

- DQN2013年バージョンとDQN2015バージョンの比較（実行条件１）<br>
![cartpole-v0_dqn2015-dqn2013_reward_episode500](https://user-images.githubusercontent.com/25688193/52899315-ffe9b680-322b-11e9-9e68-60040d8ef52f.png)<br>
![cartpole-v0_dqn2015-dqn2013_episode500](https://user-images.githubusercontent.com/25688193/52899317-02e4a700-322c-11e9-9ca0-0d20315c3224.png)<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
![cartpole-v0_dqn2015_reward_episode500](https://user-images.githubusercontent.com/25688193/52928835-15bebf00-3385-11e9-8acb-665b6f81b3b4.png)<br>
- 損失関数のグラフ（実行条件２）<br>
![cartpole-v0_dqn2015_episode500](https://user-images.githubusercontent.com/25688193/52928868-3555e780-3385-11e9-8042-ad94bee9a3eb.png)<br>
> 実行条件１より、学習が安定化していることがわかる。（実行条件１のMLPより、層数が多いため？）<br>

- DQN2013年バージョンとDQN2015バージョンの比較（実行条件２）<br>
![cartpole-v0_dqn2015-dqn2013_reward_episode500](https://user-images.githubusercontent.com/25688193/52930168-bd8abb80-338a-11e9-9add-c8b27c101dee.png)<br>
![cartpole-v0_dqn2015-dqn2013_episode500](https://user-images.githubusercontent.com/25688193/52930175-c54a6000-338a-11e9-98be-0a23635e9407.png)<br>

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>

<!--
- エピソード = 0 / 最終時間ステップ数 = 11（実行条件１）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/52898270-1c332680-321f-11e9-8b51-4f2cea030f4d.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 48<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/52898271-1c332680-321f-11e9-9c78-586fb3a40a9a.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/52898272-1ccbbd00-321f-11e9-898f-0743b8fdbc2f.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/52898273-1ccbbd00-321f-11e9-8d75-84dda4d28049.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 114<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/52898274-1ccbbd00-321f-11e9-8ec0-8d5d55e73eae.gif)<br>

- エピソード = 500 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode499](https://user-images.githubusercontent.com/25688193/52898275-1ccbbd00-321f-11e9-861f-41cecee2f610.gif)<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 11（実行条件２）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/52929008-db095680-3385-11e9-8941-de1b99cc81a1.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 53<br>
![rl_env_cartpole-v0_episode50](https://user-images.githubusercontent.com/25688193/52929009-db095680-3385-11e9-97b6-3911a99b21e3.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 168<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/52929010-dba1ed00-3385-11e9-9406-7f1fccbc7bce.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 181<br>
![rl_env_cartpole-v0_episode150](https://user-images.githubusercontent.com/25688193/52929011-dba1ed00-3385-11e9-926a-e4380331d287.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 195<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/52929013-dc3a8380-3385-11e9-8c88-15e4befd1e4d.gif)<br>

- エピソード = 250 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode250](https://user-images.githubusercontent.com/25688193/52929014-dc3a8380-3385-11e9-8d92-7860b96a4e61.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/52929015-dcd31a00-3385-11e9-8df0-4a2f2aea1843.gif)<br>

- エピソード = 350 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode350](https://user-images.githubusercontent.com/25688193/52929017-dcd31a00-3385-11e9-9077-71b716b5c786.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/52929018-dd6bb080-3385-11e9-8edd-fa51757846bf.gif)<br>

- エピソード = 450 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode450](https://user-images.githubusercontent.com/25688193/52929021-df357400-3385-11e9-9349-55e64b91a15d.gif)<br>

- エピソード = 499 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode499](https://user-images.githubusercontent.com/25688193/52929023-e066a100-3385-11e9-876a-00bfa6decb7e.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報

