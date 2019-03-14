# A2C [Advantage Actor Critic] による倒立振子課題（CartPole）【実装中...】
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法である A2C [Advantage Actor Critic] によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでの A2C のネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>

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
NUM_EPISODE = 500                   # エピソード試行回数
BRAIN_LEARNING_RATE = 0.01          # 学習率
BRAIN_GAMMDA = 0.99                 # 利得の割引率
BRAIN_KSTEP = 5                     # 先読みステップ数 k
BRAIN_LOSS_CRITIC_COEF = 0.5        # クリティック側の損失関数の重み係数
BRAIN_LOSS_ENTROPY_COEF = 0.1       # クリティック側の損失関数の重み係数
BRAIN_ADVANTAGE_SOFTPLUS = False    # アドバンテージ関数の softplus 化の有無
BRAIN_CLIPPING_MAX_GRAD = 0.5       # クリッピングする最大勾配値
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|500|
|最適化アルゴリズム|Adam|←|
|学習率：`learning_rate`|0.01|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|先読みステップ数 k `BRAIN_KSTEP`|5|←|
|クリティック側の損失関数の重み係数：`BRAIN_LOSS_CRITIC_COEF`|0.5|←|
|エントロピーの損失関数の重み係数：`BRAIN_LOSS_ENTROPY_COEF`|0.1|←|
|アドバンテージ関数の softplus 化の有無：`BRAIN_ADVANTAGE_SOFTPLUS`|`False`|`True`|
|クリッピングする最大勾配値：`BRAIN_CLIPPING_MAX_GRAD`|0.5|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|A2C のネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>アクター側の出力層：行動数（2）<br>クリティック側の出力層：1|←|


- 割引利得のエピソード毎の履歴（実行条件１）<br>

- 損失関数のグラフ（実行条件１）<br>


### ◎ コードの説明


## ■ デバッグ情報


```python
array([-0.02025148, -0.01520776,  0.020577  ,  0.03533804])

----------------------------------
CartPoleA2CBrain
<AdavantageMemory.AdavantageMemory object at 0x000001A0A6E7BC18>

_index :
 0
observations :
 tensor([[-0.0203, -0.0152,  0.0206,  0.0353],
        [-0.0289,  0.2029,  0.0028, -0.3034],
        [-0.0248,  0.0078, -0.0033, -0.0098],
        [-0.0247, -0.1873, -0.0035,  0.2818],
        [-0.0284, -0.3824,  0.0022,  0.5734],
        [-0.0361, -0.1873,  0.0136,  0.2814]])
rewards :
 tensor([[0.],
        [0.],
        [0.],
        [0.],
        [0.]])
actions :
 tensor([[1],
        [0],
        [0],
        [0],
        [1]])
done_masks :
 tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]])
total_rewards :
 tensor([[-0.0782],
        [-0.0789],
        [-0.0797],
        [-0.0806],
        [-0.0814],
        [-0.0822]])
---------------------------------
self.memory.observations[-1] : tensor([-0.0361, -0.1873,  0.0136,  0.2814])
v_function : tensor([-0.0822])
----------------------------------
actor_output : tensor([[-0.0945, -0.1613],
        [-0.0584, -0.1554],
        [-0.0905, -0.1605],
        [-0.1199, -0.1551],
        [-0.1539, -0.1596]], grad_fn=<AddmmBackward>)
critic_output : tensor([[-0.0927],
        [-0.1060],
        [-0.0953],
        [-0.0824],
        [-0.0680]], grad_fn=<AddmmBackward>)
policy : tensor([[0.5167, 0.4833],
        [0.5242, 0.4758],
        [0.5175, 0.4825],
        [0.5088, 0.4912],
        [0.5014, 0.4986]], grad_fn=<SoftmaxBackward>)
log_policy : tensor([[-0.6603, -0.7271],
        [-0.6458, -0.7429],
        [-0.6587, -0.7288],
        [-0.6757, -0.7109],
        [-0.6903, -0.6960]], grad_fn=<LogSoftmaxBackward>)
action_log_policy : tensor([[-0.7271],
        [-0.6458],
        [-0.6587],
        [-0.6757],
        [-0.6960]], grad_fn=<GatherBackward>)
loss_entropy : tensor(0.6926, grad_fn=<NegBackward>)
memory / total_reward[0:-1] tensor([[-0.0782],
        [-0.0789],
        [-0.0797],
        [-0.0806],
        [-0.0814]])
advantage : tensor([[ 0.0145],
        [ 0.0271],
        [ 0.0155],
        [ 0.0019],
        [-0.0134]], grad_fn=<SubBackward0>)
loss_actor : tensor(-0.0632, grad_fn=<SubBackward0>)
loss_critic : tensor(0.0003, grad_fn=<MeanBackward1>)
loss_fn : tensor(-0.0631, grad_fn=<AddBackward0>)

```

```python
------------------------------------
初回 rollouts.observations[step] : 
tensor([[ 0.0157,  0.0263,  0.0069, -0.0071]])

index : 0
observations :
 tensor([[[ 4.6139e-02, -2.3157e-02, -3.9206e-03, -4.9440e-02]],

        [[ 4.5676e-02, -2.1822e-01, -4.9094e-03,  2.4200e-01]],

        [[ 4.1311e-02, -2.3031e-02, -6.9320e-05, -5.2224e-02]],

        [[ 4.0851e-02, -2.1815e-01, -1.1138e-03,  2.4044e-01]],

        [[ 3.6487e-02, -4.1326e-01,  3.6949e-03,  5.3277e-01]],

        [[ 2.8222e-02, -6.0843e-01,  1.4350e-02,  8.2661e-01]]])
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
 tensor([[[0.0680]],

        [[0.0686]],

        [[0.0693]],

        [[0.0700]],

        [[0.0707]],

        [[0.0715]]])
------------------------------------

rollouts.observations[-1] : tensor([[ 0.0282, -0.6084,  0.0144,  0.8266]])
next_value : tensor([[0.0715]])

------------------------------------
state : tensor([[ 4.6139e-02, -2.3157e-02, -3.9206e-03, -4.9440e-02],
        [ 4.5676e-02, -2.1822e-01, -4.9094e-03,  2.4200e-01],
        [ 4.1311e-02, -2.3031e-02, -6.9320e-05, -5.2224e-02],
        [ 4.0851e-02, -2.1815e-01, -1.1138e-03,  2.4044e-01],
        [ 3.6487e-02, -4.1326e-01,  3.6949e-03,  5.3277e-01]])
actions tensor([[0],
        [1],
        [0],
        [0],
        [0]])
policy : tensor([[0.5362, 0.4638],
        [0.5593, 0.4407],
        [0.5360, 0.4640],
        [0.5591, 0.4409],
        [0.5815, 0.4185]], grad_fn=<SoftmaxBackward>)
log_policy : tensor([[-0.6232, -0.7684],
        [-0.5812, -0.8193],
        [-0.6236, -0.7679],
        [-0.5814, -0.8190],
        [-0.5422, -0.8710]], grad_fn=<LogSoftmaxBackward>)
action_log_policy : tensor([[-0.6232],
        [-0.8193],
        [-0.6236],
        [-0.5814],
        [-0.5422]], grad_fn=<GatherBackward>)
loss_entropy : tensor(0.6866, grad_fn=<NegBackward>)


```