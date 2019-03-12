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
NUM_EPISODE = 500               # エピソード試行回数
NUM_TIME_STEP = 200             # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.0001    # 学習率
BRAIN_GAMMDA = 0.99             # 利得の割引率
BRAIN_KSTEP = 5                 # 先読みステップ数 k
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|最適化アルゴリズム|Adam|←|
|学習率：`learning_rate`|0.0001|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|先読みステップ数 k `BRAIN_KSTEP`|5|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|←|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|←|
|A2C のネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>アクター側の出力層：行動数（2）<br>クリティック側の出力層：1|←|


- 割引利得のエピソード毎の履歴（実行条件１）<br>

- 損失関数のグラフ（実行条件１）<br>


### ◎ コードの説明


## ■ デバッグ情報

```python
CartPoleA2CBrain
<AdavantageMemory.AdavantageMemory object at 0x00000193B885B908>

_index :
 1
observations :
 tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0384, -0.0423,  0.0327,  0.0217],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
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
        [0]])
done_masks :
 tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]])
total_rewards :
 tensor([[0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.]])
----------------------------------
self._memory.observations[0:-1] : tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0384, -0.0423,  0.0327,  0.0217],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
actor_output : tensor([[-0.0918, -0.1602],
        [-0.0959, -0.1605],
        [-0.0918, -0.1602],
        [-0.0918, -0.1602],
        [-0.0918, -0.1602]], grad_fn=<AddmmBackward>)
critic_output : tensor([[-0.0945],
        [-0.0931],
        [-0.0945],
        [-0.0945],
        [-0.0945]], grad_fn=<AddmmBackward>)
policy : tensor([[0.5171, 0.4829],
        [0.5162, 0.4838],
        [0.5171, 0.4829],
        [0.5171, 0.4829],
        [0.5171, 0.4829]], grad_fn=<SoftmaxBackward>)
log_policy : tensor([[-0.6595, -0.7279],
        [-0.6614, -0.7260],
        [-0.6595, -0.7279],
        [-0.6595, -0.7279],
        [-0.6595, -0.7279]], grad_fn=<LogSoftmaxBackward>)
action_log_policy : tensor([[-0.7279],
        [-0.6614],
        [-0.6595],
        [-0.6595],
        [-0.6595]], grad_fn=<GatherBackward>)
loss_entropy : tensor(0.6926, grad_fn=<NegBackward>)
memory / total_reward[0:-1] tensor([[0.],
        [0.],
        [0.],
        [0.],
        [0.]])
advantage : tensor([[0.0945],
        [0.0931],
        [0.0945],
        [0.0945],
        [0.0945]], grad_fn=<SubBackward0>)
loss_actor : tensor(0.0565, grad_fn=<SubBackward0>)
loss_critic : tensor(0.0089, grad_fn=<MeanBackward1>)
loss_fn : tensor(0.0610, grad_fn=<AddBackward0>)
```

```python
------------------------------------
index : 0
observations :
 tensor([[[ 0.0368, -0.0013,  0.0350,  0.0368]],

        [[ 0.0368,  0.1933,  0.0358, -0.2446]],

        [[ 0.0407,  0.3879,  0.0309, -0.5258]],

        [[ 0.0484,  0.1924,  0.0204, -0.2235]],

        [[ 0.0523,  0.3872,  0.0159, -0.5097]],

        [[ 0.0600,  0.5821,  0.0057, -0.7973]]])
rewards :
 tensor([[[0.]],

        [[0.]],

        [[0.]],

        [[0.]],

        [[0.]]])
actions :
 tensor([[[1]],

        [[1]],

        [[0]],

        [[1]],

        [[1]]])
masks :
 tensor([[[1.]],

        [[1.]],

        [[1.]],

        [[1.]],

        [[1.]],

        [[1.]]])
returns :
 tensor([[[0.0268]],

        [[0.0271]],

        [[0.0274]],

        [[0.0276]],

        [[0.0279]],

        [[0.0282]]])
------------------------------------
rollouts.observations[:-1].view(-1, 4) : tensor([[ 0.0368, -0.0013,  0.0350,  0.0368],
        [ 0.0368,  0.1933,  0.0358, -0.2446],
        [ 0.0407,  0.3879,  0.0309, -0.5258],
        [ 0.0484,  0.1924,  0.0204, -0.2235],
        [ 0.0523,  0.3872,  0.0159, -0.5097]])
rollouts.actions.view(-1, 1) : tensor([[1],
        [1],
        [0],
        [1],
        [1]])
state : tensor([[ 0.0368, -0.0013,  0.0350,  0.0368],
        [ 0.0368,  0.1933,  0.0358, -0.2446],
        [ 0.0407,  0.3879,  0.0309, -0.5258],
        [ 0.0484,  0.1924,  0.0204, -0.2235],
        [ 0.0523,  0.3872,  0.0159, -0.5097]])
actions tensor([[1],
        [1],
        [0],
        [1],
        [1]])
policy : tensor([[0.4839, 0.5161],
        [0.4768, 0.5232],
        [0.4696, 0.5304],
        [0.4773, 0.5227],
        [0.4700, 0.5300]], grad_fn=<SoftmaxBackward>)
log_policy : tensor([[-0.7259, -0.6615],
        [-0.7408, -0.6477],
        [-0.7558, -0.6342],
        [-0.7396, -0.6487],
        [-0.7551, -0.6348]], grad_fn=<LogSoftmaxBackward>)
action_log_policy : tensor([[-0.6615],
        [-0.6477],
        [-0.7558],
        [-0.6487],
        [-0.6348]], grad_fn=<GatherBackward>)
loss_entropy : tensor(0.6919, grad_fn=<NegBackward>)
v_function : tensor([[[0.0726]],

        [[0.0524]],

        [[0.0404]],

        [[0.0523]],

        [[0.0398]]], grad_fn=<ViewBackward>)
memory / toatal_reward : tensor([[[0.0268]],

        [[0.0271]],

        [[0.0274]],

        [[0.0276]],

        [[0.0279]]])
advantage : tensor([[[-0.0458]],

        [[-0.0253]],

        [[-0.0130]],

        [[-0.0247]],

        [[-0.0119]]], grad_fn=<SubBackward0>)
loss_actor_gain : tensor(0.0160, grad_fn=<MeanBackward1>)
loss_critic : tensor(0.0007, grad_fn=<MeanBackward1>)
loss_total : tensor(-0.0226, grad_fn=<SubBackward0>)

```
