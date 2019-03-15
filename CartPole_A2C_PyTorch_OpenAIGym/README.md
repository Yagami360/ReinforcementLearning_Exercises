# A2C [Advantage Actor Critic] による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法である A2C [Advantage Actor Critic] によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでの A2C のネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>
※ A2C は本来は、マルチエージェントによる学習であるが、ここでの実装では簡単のため、マルチエージェントによる学習ではなく、単一のエージェントによる学習としている。<br>

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
- PyTorch : 1.0.0

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPOCHS = 2000                   # 繰り返し回数
NUM_TIME_STEP = 200                 # １エピソードの時間ステップの最大数
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

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|先読みステップ数 k `BRAIN_KSTEP`|5|←|
|最適化アルゴリズム|Adam（減衰率：beta1=0.9 beta2=0.999）|←|
|学習率：`learning_rate`|0.01|0.0001|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|クリティック側の損失関数の重み係数：`BRAIN_LOSS_CRITIC_COEF`|0.5|←|
|エントロピーの損失関数の重み係数：`BRAIN_LOSS_ENTROPY_COEF`|0.01|←|
|クリッピングする最大勾配値：`BRAIN_CLIPPING_MAX_GRAD`|0.5|←|
|A2C のネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>アクター側の出力層：行動数（2）<br>クリティック側の出力層：1|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![CartPole-v0_Reward_epoch2000_lr0 01](https://user-images.githubusercontent.com/25688193/54404650-96b86e80-4717-11e9-877e-36ccfd40bf17.png)<br>

- 損失関数のグラフ（実行条件１）<br>
![CartPole-v0_Loss_epoch2000_lr0 01](https://user-images.githubusercontent.com/25688193/54404651-96b86e80-4717-11e9-89cd-e640686caf0b.png)<br>

> 学習が不安定。<br>
> →原因は？


## ■ デバッグ情報


```python
state : tensor([ 0.0157,  0.0263,  0.0069, -0.0071])
actor_output :
 tensor([0.1522, 0.0069])
critic_output :
 tensor([0.0667])
policy : tensor([0.5363, 0.4637])
action : tensor([0])

---

state : tensor([[ 0.0157,  0.0263,  0.0069, -0.0071]])
h1  : tensor([[0.0000, 0.0000, 0.0000, 0.1131, 0.0000, 0.0000, 0.0000, 0.0068, 0.0719,
         0.0000, 0.1206, 0.0000, 0.0000, 0.4048, 0.0536, 0.0000, 0.0000, 0.0000,
         0.0180, 0.3531, 0.1242, 0.0000, 0.0000, 0.0000, 0.3661, 0.0000, 0.2135,
         0.0000, 0.0000, 0.0000, 0.0014, 0.0000]])
h2  : tensor([[0.0000, 0.0000, 0.0666, 0.1712, 0.0708, 0.0273, 0.2400, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.3104, 0.0987, 0.0000, 0.0000, 0.0000, 0.1278,
         0.0327, 0.0000, 0.1730, 0.0000, 0.0950, 0.0678, 0.1172, 0.1249, 0.0996,
         0.1055, 0.0000, 0.0203, 0.1228, 0.0962]])
actor_output : tensor([[0.1522, 0.0069]])
critic_output : tensor([[0.0667]])
policy : tensor([[0.5363, 0.4637]])
action : tensor([[0]])

```

```python
CartPoleA2CBrain
<AdavantageMemory.AdavantageMemory object at 0x00000237B821BCF8>

_index :
 0
observations :
 tensor([[ 0.0157,  0.0263,  0.0069, -0.0071],
        [ 0.0162, -0.1690,  0.0067,  0.2877],
        [ 0.0128,  0.0261,  0.0125, -0.0028],
        [ 0.0133, -0.1692,  0.0124,  0.2938],
        [ 0.0100, -0.3645,  0.0183,  0.5904],
        [ 0.0027, -0.5599,  0.0301,  0.8887]])
rewards :
 tensor([[0.],
        [0.],
        [0.],
        [0.],
        [0.]])
actions :
 tensor([[0],
        [1],
        [0],
        [0],
        [0]])
done_masks :
 tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]])
total_rewards :
 tensor([[0.0652],
        [0.0659],
        [0.0665],
        [0.0672],
        [0.0679],
        [0.0686]])
----------------------------------
actor_output : tensor([[ 0.1522,  0.0069],
        [ 0.1920, -0.0396],
        [ 0.1523,  0.0072],
        [ 0.1923, -0.0401],
        [ 0.2314, -0.0879]], grad_fn=<AddmmBackward>)
critic_output : tensor([[0.0667],
        [0.0582],
        [0.0664],
        [0.0582],
        [0.0564]], grad_fn=<AddmmBackward>)
policy : tensor([[0.5363, 0.4637],
        [0.5576, 0.4424],
        [0.5362, 0.4638],
        [0.5578, 0.4422],
        [0.5791, 0.4209]], grad_fn=<SoftmaxBackward>)
log_policy : tensor([[-0.6231, -0.7684],
        [-0.5840, -0.8156],
        [-0.6232, -0.7684],
        [-0.5837, -0.8161],
        [-0.5462, -0.8655]], grad_fn=<LogSoftmaxBackward>)
action_log_policy : tensor([[-0.6231],
        [-0.8156],
        [-0.6232],
        [-0.5837],
        [-0.5462]], grad_fn=<GatherBackward>)
loss_entropy : tensor(0.6869, grad_fn=<NegBackward>)
memory / total_reward[0:-1] tensor([[0.0652],
        [0.0659],
        [0.0665],
        [0.0672],
        [0.0679]])
advantage : tensor([[-0.0015],
        [ 0.0077],
        [ 0.0001],
        [ 0.0090],
        [ 0.0114]], grad_fn=<SubBackward0>)
loss_actor_gain : tensor(-0.0034, grad_fn=<MeanBackward1>)
loss_actor : tensor(-0.0035, grad_fn=<SubBackward0>)
loss_critic : tensor(5.4472e-05, grad_fn=<MeanBackward1>)
loss_fn : tensor(-0.0035, grad_fn=<AddBackward0>)
```

```python
index : 0
observations :
 tensor([[[ 0.0157,  0.0263,  0.0069, -0.0071]],

        [[ 0.0162, -0.1690,  0.0067,  0.2877]],

        [[ 0.0128,  0.0261,  0.0125, -0.0028]],

        [[ 0.0133, -0.1692,  0.0124,  0.2938]],

        [[ 0.0100, -0.3645,  0.0183,  0.5904]],

        [[ 0.0027, -0.5599,  0.0301,  0.8887]]])
rewards :
 tensor([[[0.]],

        [[0.]],

        [[0.]],

        [[0.]],

        [[0.]]])
actions :
 tensor([[[0]],

        [[1]],

        [[0]],

        [[0]],

        [[0]]])
masks :
 tensor([[[1.]],

        [[1.]],

        [[1.]],

        [[1.]],

        [[1.]],

        [[1.]]])
returns :
 tensor([[[0.0652]],

        [[0.0659]],

        [[0.0665]],

        [[0.0672]],

        [[0.0679]],

        [[0.0686]]])
---
loss_actor_gain : tensor(-0.0034, grad_fn=<MeanBackward1>)
loss_critic : tensor(5.4472e-05, grad_fn=<MeanBackward1>)
loss_total : tensor(-0.0035, grad_fn=<SubBackward0>)

```
