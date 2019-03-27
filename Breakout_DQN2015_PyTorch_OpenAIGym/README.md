# DQN（2015年NatureバージョンのTargetNetwork使用）によるブロック崩しゲーム（Breakout）
強化学習の学習環境用のブロック崩しゲーム（Breakout）<br>
ディープラーニングを用いた強化学習手法であるDQN [Deep Q-Network] （2015年NatureバージョンのTarget Q-Network使用）によって、Breakout を解く。<br>

※ ここでの DQN のネットワーク構成は、先の CartPole 問題での簡単のための MLP での実装はなく、本来の CNNで実装している。<br>

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
- OpenAIGym [atari]
- PyTorch : 1.0.1
- OpenCV : 4.0.0

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
DEVICE = "GPU"                          # 使用デバイス ("CPU" or "GPU")
RL_ENV = "BreakoutNoFrameskip-v0"       # 利用する atari の強化学習環境の課題名
NUM_EPISODE = 50000                     # エピソード試行回数 (Default:10000)
NUM_TIME_STEP = 1000                    # １エピソードの時間ステップの最大数
NUM_SAVE_STEP = 100                     # 強化学習環境の動画の保存間隔（単位：エピソード数）
NUM_NOOP = 30                           # エピソード開始からの何も学習しないステップ数 (Default:30)
NUM_SKIP_FRAME = 4                      # スキップするフレーム数 (Default:4)
NUM_STACK_FRAME = 4                     # モデルに一度に入力する画像データのフレーム数 (Default:4)
BRAIN_LEARNING_RATE = 0.00005           # 学習率 (Default:5e-5)
BRAIN_BATCH_SIZE = 32                   # ミニバッチサイズ (Default:32)
BRAIN_GREEDY_EPSILON_INIT = 1.0         # ε-greedy 法の ε 値の初期値 (Default:1.0)
BRAIN_GREEDY_EPSILON_FINAL = 0.01       # ε-greedy 法の ε 値の最終値 (Default:0.1)
BRAIN_GREEDY_EPSILON_STEPS = 1000000    # ε-greedy 法の ε が減少していくフレーム数　(Default:1_000_000)
BRAIN_GAMMDA = 0.99                     # 利得の割引率 (Default:0.99)
BRAIN_FREC_TARGET_UPDATE = 1000         # Target Network との同期頻度（Default:10_000） 
MEMORY_CAPACITY = 10000                 # Experience Relay 用の学習用データセットのメモリの最大の長さ (Default:1_000_000)
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|5,000|10,000|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|1000|←|
|エピソード開始からの何も学習しないステップ数：`NUM_NOOP`|30|10|
|モデルに一度に入力する画像データのフレーム数：`NUM_STACK_FRAME`|4|←|
|スキップするフレーム数：`NUM_SKIP_FRAME`|4|←|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|←|
|最適化アルゴリズム|RMSprop<br>減衰項：デフォルト値|←|
|学習率：`learning_rate`|0.00005|0.0005|
|損失関数|smooth L1 関数（＝Huber 関数）|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON_INIT`|1.0|←|
|ε-greedy 法の ε 値の最終値：`BRAIN_GREEDY_EPSILON_FINAL`|0.01|←|
|ε-greedy 法の減衰ステップ数：`BRAIN_GREEDY_EPSILON_STEPS`|50,000|100,00|
|Target Network との同期頻度：`BRAIN_FREC_TARGET_UPDATE`|1000|←|
|Experience Relay用のメモリサイズ：`MEMORY_CAPACITY`|10,000|←|
|報酬の設定|Breakout のデフォルト報酬<br>・下段の青色＆緑色のブロック崩し：１点<br>・中央の黄色＆黄土色のブロック崩し：４点<br>・上段のオレンジ＆赤色のブロック崩し：７点<br>に対して、符号化関数 sign で 0.0 or 1.0 の範囲にクリッピング|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|DQNのネットワーク構成|CNN<br>(0): Conv2d(in_channels=**1**, out_channels=32, kernel_size=(8, 8), stride=(4, 4))<br>(1): ReLU()<br>(2): Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))<br>(3): ReLU()<br>(4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))<br>(5): ReLU()<br>(6): Flatten()<br>(7): Linear(in_features=7×7×64, out_features=512, bias=True)<br>(8): ReLU()<br>Linear(in_features=512, out_features=**4**, bias=True)|←|

<!--
|報酬の設定|Breakout のデフォルト報酬<br>・下段の青色＆緑色のブロック崩し：１点<br>・中央の黄色＆黄土色のブロック崩し：４点<br>・上段のオレンジ＆赤色のブロック崩し：７点<br>に対して0.0~1.0の範囲にクリッピング|←|
-->

<br>

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![BreakoutNoFrameskip-v0_DQN2015_Reward_episode5000_ts1000_lr5e-05_noop30](https://user-images.githubusercontent.com/25688193/55045104-074c7d00-5080-11e9-9dd5-9fc364507843.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![BreakoutNoFrameskip-v0_DQN2015_Loss_episode5000_ts1000_lr5e-05_noop30](https://user-images.githubusercontent.com/25688193/55045103-06b3e680-5080-11e9-89a6-901b7c4eb003.png)<br>

<br>

以下のアニメーションは、Breakout のブロック崩しを行う様子を示したアニメーションである。<br>
<!--
エピソードの経過と共に、徐々にブロック崩しが出来るようになっており、徐々に学習できていることがわかる。<br>
-->


- エピソード = 0 / 最終時間ステップ数 = 24 / total_reward = 0.0（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode0](https://user-images.githubusercontent.com/25688193/54999042-c9fdd600-5012-11e9-97af-ae7153c8f860.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = xx / total_reward = x.x（実行条件１）<br>
<br>

- エピソード = 500 / 最終時間ステップ数 = xx / total_reward = x.x（実行条件１）<br>
<br>

- エピソード = 1000 / 最終時間ステップ数 = xx / total_reward = x.x（実行条件１）<br>
<br>

- エピソード = 4400 / 最終時間ステップ数 = 139 / total_reward = x.x（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode4400](https://user-images.githubusercontent.com/25688193/55045191-52ff2680-5080-11e9-9e80-42528bd09a72.gif)<br>

- エピソード = 5000 / 最終時間ステップ数 = 80 / total_reward = x.x（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode4999](https://user-images.githubusercontent.com/25688193/55045151-3a8f0c00-5080-11e9-9573-fdadac98fa26.gif)<br>

- エピソード = 10000 / 最終時間ステップ数 = xx / total_reward = x.x（実行条件１）<br>
<br>

<!--
- エピソード = 0 / 最終時間ステップ数 = 40（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode0](https://user-images.githubusercontent.com/25688193/54874023-86df0f80-4e26-11e9-831a-7ceb512b0b64.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 28（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode100](https://user-images.githubusercontent.com/25688193/54874024-86df0f80-4e26-11e9-8e24-84d8937d20a6.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 28（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode200](https://user-images.githubusercontent.com/25688193/54874025-8777a600-4e26-11e9-8fdb-fea9a0925d04.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 56（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode300](https://user-images.githubusercontent.com/25688193/54874026-8777a600-4e26-11e9-89e3-804e7f3350f9.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 40（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode400](https://user-images.githubusercontent.com/25688193/54874027-8777a600-4e26-11e9-97b7-61d84bbd3bb8.gif)<br>

- エピソード = 500 / 最終時間ステップ数 = 28（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode500](https://user-images.githubusercontent.com/25688193/54874028-8777a600-4e26-11e9-86f8-bb25db9d04db.gif)<br>

- エピソード = 1000 / 最終時間ステップ数 = 70（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode1000](https://user-images.githubusercontent.com/25688193/54874032-8b0b2d00-4e26-11e9-9689-6149fed10100.gif)<br>

- エピソード = 2000 / 最終時間ステップ数 = 24<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode2000](https://user-images.githubusercontent.com/25688193/54874071-459b2f80-4e27-11e9-826a-9517a7b15d3f.gif)<br>

- エピソード = 3000 / 最終時間ステップ数 = 64<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode3000](https://user-images.githubusercontent.com/25688193/54874073-4764f300-4e27-11e9-9c0b-05c44cbf5ce6.gif)<br>

- エピソード = 4000 / 最終時間ステップ数 = 100<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode4000](https://user-images.githubusercontent.com/25688193/54874074-49c74d00-4e27-11e9-9728-fc0ced3d0c3e.gif)<br>

- エピソード = 9000 / 最終時間ステップ数 = 107<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode9000](https://user-images.githubusercontent.com/25688193/54874087-7d09dc00-4e27-11e9-950a-b283eb564650.gif)<br>

- エピソード = 9100 / 最終時間ステップ数 = 172<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode9100](https://user-images.githubusercontent.com/25688193/54874102-c2c6a480-4e27-11e9-9295-04ce4f1282ae.gif)<br>

- エピソード = 10000 / 最終時間ステップ数 = 76<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode9999](https://user-images.githubusercontent.com/25688193/54874083-682d4880-4e27-11e9-82a6-001054c41210.gif)<br>
-->

<!--
- エピソード = 100 / 最終時間ステップ数 = 56（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode100](https://user-images.githubusercontent.com/25688193/54818179-057b6600-4cdc-11e9-930b-4de997330f4b.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 99（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode200](https://user-images.githubusercontent.com/25688193/54818281-378cc800-4cdc-11e9-8dc2-3ba7a7fa36c0.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 132（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode400](https://user-images.githubusercontent.com/25688193/54818365-6dca4780-4cdc-11e9-999a-fa7563fa072e.gif)<br>

- エピソード = 500 / 最終時間ステップ数 = 22（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode500](https://user-images.githubusercontent.com/25688193/54818180-057b6600-4cdc-11e9-9be0-61a466710ce1.gif)<br>

- エピソード = 800 / 最終時間ステップ数 = 1092（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode800](https://user-images.githubusercontent.com/25688193/54818421-8dfa0680-4cdc-11e9-90cd-262651961a46.gif)<br>
> 動いていないのに、継続ステップ数が大きい？

- エピソード = 900 / 最終時間ステップ数 = 124（実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode900](https://user-images.githubusercontent.com/25688193/54818587-f517bb00-4cdc-11e9-98c5-04a728f442e3.gif)<br>
> 動いていない？

- エピソード = 1000 / 最終時間ステップ数 = （実行条件１）<br>
![RL_ENV_BreakoutNoFrameskip-v0_Episode999](https://user-images.githubusercontent.com/25688193/54818182-057b6600-4cdc-11e9-8627-cdc190cb1a45.gif)<br>
-->