# DQN（2015年NatureバージョンのTargetNetwork使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2015年NatureバージョンのTarget Q-Network使用）によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDQNのネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Deep Q Network（DQN）](http://yagami12.hatenablog.com/entry/2019/02/22/210608#DeepQNetwork)


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
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5（減衰）|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>出力層：行動数（2）|MLP（4層）<br>入力層：状態数（4）<br>隠れ層１：32ノード<br>隠れ層２：32ノード<br>出力層：行動数（2）|


<!--
転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|
-->

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_dqn2015_reward_mlp3_episode500](https://user-images.githubusercontent.com/25688193/53780206-a5986780-3f46-11e9-8f1b-1e3f597f77b5.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![cartpole-v0_dqn2015_mlp3_episode500](https://user-images.githubusercontent.com/25688193/53780207-a5986780-3f46-11e9-8d3d-47254bf568c6.png)<br>
> DQN2013バージョンより、学習が安定していることがわかる。（Target Q-Network 同期による学習安定化効果？）

<!--
- DQN2013年バージョンとDQN2015バージョンの比較（実行条件１）<br>
![cartpole-v0_dqn2015-dqn2013_mlp3_reward_episode500](https://user-images.githubusercontent.com/25688193/53071607-0c208d00-3526-11e9-9fbf-a01e4deb7d0a.png)<br>
![cartpole-v0_dqn2015-dqn2013_mlp3_episode500](https://user-images.githubusercontent.com/25688193/53071608-0cb92380-3526-11e9-875f-0d7065c57c35.png)<br>
-->

<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
<!--
![cartpole-v0_dqn2015_reward_episode500](https://user-images.githubusercontent.com/25688193/52928835-15bebf00-3385-11e9-8acb-665b6f81b3b4.png)<br>
-->
<!--
![cartpole-v0_dqn2015_reward_mlp4_episode500](https://user-images.githubusercontent.com/25688193/53069071-0cb52580-351e-11e9-9b99-602261330fce.png)<br>
-->
![cartpole-v0_dqn2015_reward_mlp4_episode500](https://user-images.githubusercontent.com/25688193/53780902-cc0bd200-3f49-11e9-8235-f9168242d9cb.png)<br>

- 損失関数のグラフ（実行条件２）<br>
<!--
![cartpole-v0_dqn2015_episode500](https://user-images.githubusercontent.com/25688193/52928868-3555e780-3385-11e9-8042-ad94bee9a3eb.png)<br>
-->
<!--
![cartpole-v0_dqn2015_mlp4_episode500](https://user-images.githubusercontent.com/25688193/53069072-0cb52580-351e-11e9-87c8-71dc2d3fd948.png)<br>
-->
![cartpole-v0_dqn2015_mlp4_episode500](https://user-images.githubusercontent.com/25688193/53780901-c9a97800-3f49-11e9-8587-f2eae79bd4a6.png)<br>

<!--
- DQN2013年バージョンとDQN2015バージョンの比較（実行条件２）<br>
![cartpole-v0_dqn2015-dqn2013_mlp4 reward_episode500](https://user-images.githubusercontent.com/25688193/53072115-8bfb2700-3527-11e9-91ee-f970cd331344.png)<br>
![cartpole-v0_dqn2015-dqn2013_mlp4_episode500](https://user-images.githubusercontent.com/25688193/53072117-8c93bd80-3527-11e9-9237-2e27ff2a4e6f.png)<br>
-->

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>


- エピソード = 0 / 最終時間ステップ数 = 10（実行条件２）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/53780992-391f6780-3f4a-11e9-8a8c-f15c0bce8b4b.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 14<br>
![rl_env_cartpole-v0_episode50](https://user-images.githubusercontent.com/25688193/53780995-3de41b80-3f4a-11e9-8a9c-17a748f8cdd2.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 63<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/53780997-40df0c00-3f4a-11e9-9dfa-9e961126f4c6.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 198<br>
![rl_env_cartpole-v0_episode150](https://user-images.githubusercontent.com/25688193/53781002-450b2980-3f4a-11e9-9177-18131ee7740f.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/53781058-84397a80-3f4a-11e9-9806-2548b9b807b8.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/53781061-869bd480-3f4a-11e9-8864-9b213b7fedfb.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/53781066-8a2f5b80-3f4a-11e9-9d36-0cb1e1fafacf.gif)<br>

- エピソード = 499 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode499](https://user-images.githubusercontent.com/25688193/53781068-8bf91f00-3f4a-11e9-9551-9dad68adc5dc.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報

