# Q学習による倒立振子課題（CartPole）
強化学習の学習環境用の倒立振子課題 CartPole。<br>
方策オフ型TD制御アルゴリズムである Q学習 によって、単純な２次元の倒立振子課題を解く。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論
    1. [【外部リンク】強化学習 / Q学習](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92.md#Q%E5%AD%A6%E7%BF%92)


## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- OpenAIGym : 0.10.9

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
NUM_EPISODE = 500           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
NUM_DIZITIZED = 6           # 各状態の離散値への分割数
BRAIN_LEARNING_RATE = 0.5   # 学習率
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99         # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|エピソード試行回数：`NUM_EPISODE`|500|500|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|←|
|各状態の離散値への分割数：`NUM_DIZITIZED`|6|←|
|学習率：`learning_rate`|0.5|←|
|利得の割引率：`BRAIN_GAMMDA`|0.99|←|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|←|
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0に設定|転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+1|

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_qlearning_reward_episode500](https://user-images.githubusercontent.com/25688193/52897552-5c8fa600-3219-11e9-87e4-ba7440f9608c.png)<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
![cartpole-v0_qlearning_reward_episode500](https://user-images.githubusercontent.com/25688193/52897160-a7f38580-3214-11e9-8db5-c1ba4f53dccb.png)<br>

<br>

<!--
以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 24（実行条件１）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/52897533-176b7400-3219-11e9-93bd-e913548b16cf.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 85<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/52897531-176b7400-3219-11e9-8ca7-029ebdd4cf7d.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 141<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/52897540-294d1700-3219-11e9-9a53-4b00c75e2dff.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/52897543-3964f680-3219-11e9-9b37-ffddd4166919.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 21<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/52897549-51d51100-3219-11e9-9900-8aa4088f24cd.gif)<br>

- エピソード = 499 / 最終時間ステップ数 = 158<br>
![rl_env_cartpole-v0_episode499](https://user-images.githubusercontent.com/25688193/52897557-6f09df80-3219-11e9-8538-4e03edb05e64.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報
```python


```