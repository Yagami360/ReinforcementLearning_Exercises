# A2C [Advantage Actor Critic] による CartPole（シングルスレッドでの非分散学習）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法である A2C [Advantage Actor Critic] によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでの A2C のネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>
※ 本来の A2C は、マルチスレッドによる分散学習で実装されるものであるが、ここでは簡単のため、シングルスレッドによる非分散学習で実装している。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / A2C [Advantage Actor-Critic]](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#A2C)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9
- PyTorch : 1.0.1

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
DEVICE = "GPU"                      # 使用デバイス ("CPU" or "GPU")
NUM_EPOCHS = 2000                   # 繰り返し回数
NUM_TIME_STEP = 200                 # １エピソードの時間ステップの最大数
NUM_SAVE_STEP = 250                 # 強化学習環境の動画の保存間隔（単位：エピソード数）
NUM_KSTEP = 5                       # 先読みステップ数 k
BRAIN_LEARNING_RATE = 0.01          # 学習率
BRAIN_GAMMDA = 0.99                 # 利得の割引率
BRAIN_KSTEP = 5                     # 先読みステップ数 k
BRAIN_LOSS_CRITIC_COEF = 0.5        # クリティック側の損失関数の重み係数
BRAIN_LOSS_ENTROPY_COEF = 0.01      # クリティック側の損失関数の重み係数
BRAIN_CLIPPING_MAX_GRAD = 0.5       # クリッピングする最大勾配値
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|値（実行条件３）|
|---|---|---|---|
|エポック数：`NUM_EPOCHS`|2000|50000|←|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|←|
|先読みステップ数 k `BRAIN_KSTEP`|5|←|2|
|最適化アルゴリズム|Adam（減衰率：beta1=0.9 beta2=0.999）|←|←|
|学習率：`learning_rate`|0.01|0.0001|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|クリティック側の損失関数の重み係数：`BRAIN_LOSS_CRITIC_COEF`|0.5|←|
|エントロピーの損失関数の重み係数：`BRAIN_LOSS_ENTROPY_COEF`|0.01|←|
|クリッピングする最大勾配値：`BRAIN_CLIPPING_MAX_GRAD`|0.5|←|←|
|A2C のネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>アクター側の出力層：行動数（2）<br>クリティック側の出力層：1|←|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|←|

- 割引利得のエピソード毎の履歴（実行条件１）<br>
<!--
![CartPole-v0_Reward_epoch2000_lr0 01](https://user-images.githubusercontent.com/25688193/54404650-96b86e80-4717-11e9-877e-36ccfd40bf17.png)<br>
-->
![CartPole-v0_Reward_epoch2000_lr0 01_maxgrad0 5](https://user-images.githubusercontent.com/25688193/55305716-8c72d000-548c-11e9-9a35-d422cda66512.png)<br>

- 損失関数のグラフ（実行条件１）<br>
<!--
![CartPole-v0_Loss_epoch2000_lr0 01](https://user-images.githubusercontent.com/25688193/54404651-96b86e80-4717-11e9-89cd-e640686caf0b.png)<br>
-->
![CartPole-v0_Loss_epoch2000_lr0 01_maxgrad0 5](https://user-images.githubusercontent.com/25688193/55305717-8d0b6680-548c-11e9-842b-c030de65ae43.png)<br>

> 学習が不安定。<br>
> →学習回数が足りてない？学習率が大きすぎて最適解を飛び越えている？<br>

<!--
- 割引利得のエピソード毎の履歴（実行条件x）<br>
![CartPole-v0_Reward_epoch50000_lr0 001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54407990-89a17c80-4723-11e9-8dd4-ce939958bf2a.png)<br>

- 損失関数のグラフ（実行条件x）<br>
![CartPole-v0_Loss_epoch50000_lr0 001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54408040-a1790080-4723-11e9-99c4-b4606773f7bb.png)<br>

> 途中からうまく学習できていない。
-->

- 割引利得のエピソード毎の履歴（実行条件２）<br>
<!--
![CartPole-v0_Reward_epoch50000_lr0 0001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54408248-7cd15880-4724-11e9-872b-19a0652cd538.png)<br>
-->
![CartPole-v0_Reward_epoch50000_lr0 0001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/55309707-8cc59800-5499-11e9-9765-88d3ace7cbbf.png)<br>

> 実行条件１の学習率より、高い報酬に到達できている。<br>

- 損失関数のグラフ（実行条件２）<br>
<!--
![CartPole-v0_Loss_epoch50000_lr0 0001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54408251-7fcc4900-4724-11e9-8aa1-ae3b70e65ced.png)<br>
-->
![CartPole-v0_Loss_epoch50000_lr0 0001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/55309711-8e8f5b80-5499-11e9-8650-981d478af981.png)<br>

<!--
- 割引利得のエピソード毎の履歴（実行条件３）<br>
![CartPole-v0_Reward_epoch50000_lr1e-05_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54409274-48ac6680-4729-11e9-9534-44c2b1401d57.png)

- 損失関数のグラフ（実行条件３）<br>
![CartPole-v0_Loss_epoch50000_lr1e-05_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54409276-4c3fed80-4729-11e9-95c6-30529b743799.png)
-->

<!--
- 割引利得のエピソード毎の履歴（実行条件３）<br>
![CartPole-v0_Reward_epoch50000_lr0 0001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54409481-48609b00-472a-11e9-878c-22e3cc744392.png)

- 損失関数のグラフ（実行条件３）<br>
![CartPole-v0_Loss_epoch50000_lr0 0001_maxgrad0 5](https://user-images.githubusercontent.com/25688193/54409487-4eef1280-472a-11e9-8f9e-f600ac54a2a2.png)
-->