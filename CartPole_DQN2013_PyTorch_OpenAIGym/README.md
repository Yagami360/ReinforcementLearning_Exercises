# DQN（2013年バージョンのTargetNetwork非使用）による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2013年バージョンのTarget Q-Network非使用）によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDQNのネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。

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
|エピソード試行回数：`NUM_EPISODE`|500|←|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|学習率：`learning_rate`|0.0001|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|Adam|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10000|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0に設定|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|MLP（３層）<br>入力層：状態数<br>隠れ層：32ノード<br>出力層：行動数|MLP（４層）<br>入力層：状態数<br>隠れ層１：32ノード<br>隠れ層２：32ノード<br>出力層：行動数|

<!--
転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|
-->

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_dqn2013_reward_episode500](https://user-images.githubusercontent.com/25688193/53067895-90b8de80-3519-11e9-982e-a027e512fafa.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![cartpole-v0_dqn2013_episode500](https://user-images.githubusercontent.com/25688193/53067890-8ac2fd80-3519-11e9-8117-925c7e4f9d92.png)<br>
> 途中で損失関数の値が発散しており、その後０付近の収束しておらず、うまく学習できていないことがわかる。<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
![cartpole-v0_dqn2013_reward_episode500](https://user-images.githubusercontent.com/25688193/53067446-8eee1b80-3517-11e9-882a-5d32b68e3468.png)<br>

- 損失関数のグラフ（実行条件２）<br>
![cartpole-v0_dqn2013_episode500](https://user-images.githubusercontent.com/25688193/53067444-8c8bc180-3517-11e9-9efd-d07609072ca3.png)<br>
> エピソードが経過するにつれて、損失関数の値が０付近の値に向かって収束しており、うまく学習できていることがわかる。<br>
> また、実行条件１より、学習が安定化していることがわかる。（実行条件１のMLPより、層数が多いため？）<br>


<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
<!--
※ ポールを左右に振りながらバランスを取るときの振り幅が、Q学習や Sarsa では大きかったのに対して、この DQN では小さい傾向がある？<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 10（実行条件２）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/53067035-bd6af700-3515-11e9-8f05-2be510a31487.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 15<br>
![rl_env_cartpole-v0_episode50](https://user-images.githubusercontent.com/25688193/53067037-be038d80-3515-11e9-9020-47436bea93c0.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 14<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/53067038-be9c2400-3515-11e9-9547-872a4a59cfad.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 20<br>
![rl_env_cartpole-v0_episode150](https://user-images.githubusercontent.com/25688193/53067031-bcd26080-3515-11e9-8651-92b9f726da33.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/53067032-bcd26080-3515-11e9-940d-9b3da2dec366.gif)<br>

- エピソード = 250 / 最終時間ステップ数 = 187<br>
![rl_env_cartpole-v0_episode250](https://user-images.githubusercontent.com/25688193/53067034-bd6af700-3515-11e9-85d2-f40d8b03705d.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 133<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/53067076-ef7c5900-3515-11e9-9a46-5eba1861e3cc.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/53067258-cad4b100-3516-11e9-953e-b27132242d61.gif)<br>

- エピソード = 500 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode499](https://user-images.githubusercontent.com/25688193/53067451-91e90c00-3517-11e9-9872-31a85674f57a.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報

