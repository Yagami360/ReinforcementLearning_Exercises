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
NUM_EPISODE = 250           # エピソード試行回数
NUM_TIME_STEP = 200         # １エピソードの時間ステップの最大数
NUM_DIZITIZED = 6           # 各状態の離散値への分割数
BRAIN_LEARNING_RATE = 0.5   # 学習率
BRAIN_GREEDY_EPSILON = 0.5  # ε-greedy 法の ε 値
BRAIN_GAMMDA = 0.99         # 割引率
```

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|
|---|---|
|エピソード試行回数：`NUM_EPISODE`|250|
|１エピソードの時間ステップの最大数：`NUM_TIME_STEP`|200|
|各状態の離散値への分割数：`NUM_DIZITIZED`|6|
|学習率：`learning_rate`|0.5|
|利得の割引率：`BRAIN_GAMMDA`|0.99|
|ε-greedy 法の ε 値の初期値：`BRAIN_GREEDY_EPSILON`|0.5|

<!--
![rl_env_cartpole-v0](https://user-images.githubusercontent.com/25688193/51795542-10f46880-2228-11e9-9e62-87b889cdb53a.gif)<br>
-->

<br>

以下のアニメーションは、CarPole のポールのバランスを取る様子を示したアニメーションである。<br>
エピソードの経過と共に、うまくバランスが取れるようになっており、うまく学習できていることがわかる。<br>

- エピソード = 0 / 最終時間ステップ数 = 21<br>
![rl_env_cartpole-v0_qlearning_episode0](https://user-images.githubusercontent.com/25688193/52696314-11c02500-2fb2-11e9-9ce7-5d3e2b5ea82f.gif)<br>

- エピソード = 10 / 最終時間ステップ数 = 27
![rl_env_cartpole-v0_qlearning_episode10](https://user-images.githubusercontent.com/25688193/52696328-1dabe700-2fb2-11e9-9826-a763790db093.gif)<br>

- エピソード = 20 / 最終時間ステップ数 = 9<br>
![rl_env_cartpole-v0_qlearning_episode20](https://user-images.githubusercontent.com/25688193/52696499-7e3b2400-2fb2-11e9-9627-8586d2a2d624.gif)<br>

- エピソード = 50 / 最終時間ステップ数 = 34<br>
![rl_env_cartpole-v0_qlearning_episode50](https://user-images.githubusercontent.com/25688193/52696530-8a26e600-2fb2-11e9-8102-3aea103702e4.gif)<br>

- エピソード = 60 / 最終時間ステップ数 = 177<br>
![rl_env_cartpole-v0_qlearning_episode60](https://user-images.githubusercontent.com/25688193/52696541-927f2100-2fb2-11e9-9f4c-00ee8dad7e10.gif)<br>

- エピソード = 70 / 最終時間ステップ数 = 169<br>
![rl_env_cartpole-v0_qlearning_episode70](https://user-images.githubusercontent.com/25688193/52696549-99a62f00-2fb2-11e9-9d35-4d3fa2a7f73f.gif)<br>

- エピソード = 80 / 最終時間ステップ数 = 64<br>
![rl_env_cartpole-v0_qlearning_episode80](https://user-images.githubusercontent.com/25688193/52696574-a88ce180-2fb2-11e9-97c3-3b055a01fc8f.gif)<br>

- エピソード = 90 / 最終時間ステップ数 = 90<br>
![rl_env_cartpole-v0_qlearning_episode90](https://user-images.githubusercontent.com/25688193/52696595-b2aee000-2fb2-11e9-8fb4-687781694fb2.gif)<br>

- エピソード = 100 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_qlearning_episode100](https://user-images.githubusercontent.com/25688193/52696608-bd697500-2fb2-11e9-9f0d-24cae2042e39.gif)<br>

- エピソード = 110 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_qlearning_episode110](https://user-images.githubusercontent.com/25688193/52696677-ebe75000-2fb2-11e9-97a5-8650e2c932a9.gif)<br>

- エピソード = 150 / 最終時間ステップ数 = 199<br>
![rl_env_cartpole-v0_qlearning_episode150](https://user-images.githubusercontent.com/25688193/52696769-2cdf6480-2fb3-11e9-86e5-00d8dced652f.gif)<br>

- エピソード = 190 / 最終時間ステップ数 = 151<br>
![rl_env_cartpole-v0_qlearning_episode190](https://user-images.githubusercontent.com/25688193/52697062-e50d0d00-2fb3-11e9-881b-dbacbd29e101.gif)<br>

- エピソード = 200 / 最終時間ステップ数 = 199
![rl_env_cartpole-v0_qlearning_episode200](https://user-images.githubusercontent.com/25688193/52697029-d9214b00-2fb3-11e9-9b3a-44c455ca0f8a.gif)<br>


### ◎ コードの説明


## ■ デバッグ情報
```python


```