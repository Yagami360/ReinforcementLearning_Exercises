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
|報酬の設定|転倒：-1<br>連続 `NUM_TIME_STEP=200`回成功：+1<br>それ以外：0|転倒：-1<br>連続 `NUM_TIME_STEP`回成功：+`NUM_TIME_STEP=200`<br>それ以外：+0.01|
|シード値|`np.random.seed(8)`<br>`random.seed(8)`<br>`env.seed(8)`|←|

- 割引利得のエピソード毎の履歴（実行条件１）<br>
![cartpole-v0_qlearning_reward_episode500](https://user-images.githubusercontent.com/25688193/53066078-990d1b80-3511-11e9-974d-eb456fa16dfc.png)<br>

- 割引利得のエピソード毎の履歴（実行条件２）<br>
<!--
![cartpole-v0_qlearning_reward_episode500](https://user-images.githubusercontent.com/25688193/52897160-a7f38580-3214-11e9-8db5-c1ba4f53dccb.png)<br>
-->

<br>


以下のアニメーションは、Q学習によって、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
<!--
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>
-->

- エピソード = 0 / 最終時間ステップ数 = 17（実行条件１）<br>
![rl_env_cartpole-v0_episode0](https://user-images.githubusercontent.com/25688193/53065974-3e73bf80-3511-11e9-9465-c85ffc499bc4.gif)<br>

- エピソード = 10 / 最終時間ステップ数 = 46<br>
![rl_env_cartpole-v0_episode10](https://user-images.githubusercontent.com/25688193/53066173-fa34ef00-3511-11e9-94de-328636364d58.gif)<br>

- エピソード = 20 / 最終時間ステップ数 = 32<br>
![rl_env_cartpole-v0_episode20](https://user-images.githubusercontent.com/25688193/53066174-facd8580-3511-11e9-9f17-603435deda05.gif)<br>

- エピソード = 30 / 最終時間ステップ数 = 38<br>
![rl_env_cartpole-v0_episode30](https://user-images.githubusercontent.com/25688193/53066175-facd8580-3511-11e9-9568-5bd357c04f1f.gif)<br>

- エピソード = 40 / 最終時間ステップ数 = 85<br>
![rl_env_cartpole-v0_episode40](https://user-images.githubusercontent.com/25688193/53066177-facd8580-3511-11e9-82a7-dceeb57262ce.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 33<br>
![rl_env_cartpole-v0_episode50](https://user-images.githubusercontent.com/25688193/53066178-fb661c00-3511-11e9-95d2-bded651f8b98.gif)<br>

- エピソード = 60 / 最終時間ステップ数 = 31<br>
![rl_env_cartpole-v0_episode60](https://user-images.githubusercontent.com/25688193/53066318-92cb6f00-3512-11e9-938e-9dcb55156b9f.gif)<br>

- エピソード = 70 / 最終時間ステップ数 = 101<br>
![rl_env_cartpole-v0_episode70](https://user-images.githubusercontent.com/25688193/53066242-3700e600-3512-11e9-99e4-f6d5a6aca805.gif)<br>

- エピソード = 80 / 最終時間ステップ数 = 199<br>
[rl_env_cartpole-v0_episode80](https://user-images.githubusercontent.com/25688193/53066243-3700e600-3512-11e9-9233-257106b8cc2e.gif)<br>

- エピソード = 90 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode90](https://user-images.githubusercontent.com/25688193/53066319-93fc9c00-3512-11e9-940b-154d2f1fb1d2.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode100](https://user-images.githubusercontent.com/25688193/53065975-3e73bf80-3511-11e9-9f49-9407e3f03394.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode200](https://user-images.githubusercontent.com/25688193/53065976-3f0c5600-3511-11e9-9215-ee28da66bf47.gif)<br>

- エピソード = 300 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode300](https://user-images.githubusercontent.com/25688193/53065978-3f0c5600-3511-11e9-8ac6-4bfa277caf5b.gif)<br>

- エピソード = 400 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_episode400](https://user-images.githubusercontent.com/25688193/53066050-80046a80-3511-11e9-8377-2cb7fefb16c6.gif)<br>

- エピソード = 499 / 最終時間ステップ数 = 151<br>
![rl_env_cartpole-v0_episode499](https://user-images.githubusercontent.com/25688193/53066101-b04c0900-3511-11e9-91c2-4673d9b83783.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報
```python


```