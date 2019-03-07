# DDQN & Prioritized Experience Replay による倒立振子課題（CartPole）【実装中...】
強化学習の学習環境用の倒立振子課題 CartPole。<br>
ディープラーニングを用いた強化学習手法であるDDQN [Double-DQN] をベースに、Prioritized Experience Replay による Relpay Memory によって、単純な２次元の倒立振子課題を解く。<br>

※ ここでのDDQNのネットワーク構成は、簡単のため、CNNではなく多層パーセプトロン（MLP）で代用したもので実装している。<br>
※ ここでの Prioritized Experience Replay での Replay Memory のデータ構造は、簡単のため、リスト（deque）で実装している。<br>
※ また、簡単のため、サンプリングの確率分布は Stochastic sampling method で、優先度は Proportional Prioritization ( direct ) で実装している。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Prioritized Experience Replay（優先順位付き経験再生）](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#PrioritizedExperienceReplay)


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
NUM_TIME_STEP = 200                 # １エピソードの時間ステップの最大数
BRAIN_LEARNING_RATE = 0.0001        # 学習率
BRAIN_BATCH_SIZE = 32               # ミニバッチサイズ
BRAIN_GREEDY_EPSILON = 0.5          # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99                 # 利得の割引率
MEMORY_CAPACITY = 10000             # Relay Memory のメモリの最大の長さ
MEMORY_TD_ERROR_EPSILON = 0.0001    # サンプリング優先度を計算するTD誤差のバイアス値
MEMORY_CHANGE_EPISODE = 30          # Experience Replay → PrioritizedExperienceReplay に切り替えるエピソード数
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|
|学習率：`learning_rate`|0.0001|
|ミニバッチサイズ：`BRAIN_LEARNING_RATE`|32|
|最適化アルゴリズム|Adam|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5（減衰）|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`torch.manual_seed(8)`<br>`env.seed(8)`|
|DQNのネットワーク構成|MLP（3層）<br>入力層：状態数（4）<br>隠れ層：32ノード<br>出力層：行動数（2）|
|Replay Memory のメモリサイズ：`MEMORY_CAPACITY`|10000|
|Replay Memory のデータ構造|リスト（deque）|
|サンプリングの確率分布<br>Stochastic sampling method / Proportional Prioritization<br>バイアス値：`MEMORY_TD_ERROR_EPSILON`|0.0001|
|サンプリング方法|層状抽出法？|
|Experience Replay → PrioritizedExperienceReplay に切り替えるエピソード数：`MEMORY_CHANGE_EPISODE`|30|

- 割引利得のエピソード毎の履歴（実行条件１）<br>
<br>

- 損失関数のグラフ（実行条件１）<br>
<br>


<br>

### ◎ コードの説明


## ■ デバッグ情報

```python

```