# 強化学習と人間の模倣学習によるTorcs運転エージェントの融合モデル

# インストール
## インストールの前
  - インターネット接続が重要である
  - ハードディスクには2ギガ以上
  - sudo 権限
`install_script.sh`はUbuntuを対応する依存性をインストールスクリプトである。

実行する前は、手動にAnacondaがインストールする必要がある.
`source install_script.sh`を実行すると、以下のものが設定される:
  - Torcsのバイナリー（本プログラム）と依存性
  - Anacondaのバーチャル環境: baselines-torcs

うまくいかなければ、エラーによってスクリプトの内容を変更し、また手動各コマンドを実行するのは可能。

重要なenvironment variablesを以下のように設定できる。
```
export TORCS_DATA_DIR="/home/$USER/torcs_data/" # プレイヤデータ収集ようにフォルダ
export DISPLAY=:1 # xvfb-runでTorcsプログラムを示せず学習する
```

# 学習
以前に作成されたAnacondaおバーチャル緩急を機動する
```
conda activate baselines-torcs
```

## DDPG
```
python -m baselines.torcs_ddpg.main
```
結果は ./baselines/torcs_ddpg/resultに書き込まれる

## GAIL
```
python -m baselines.gail.run_mujoco
```
結果は ./baselines/gail/resultに書き込まれる

# Hybrid (Remi)
```
python -m baselines.remi.run_mujoco

```
結果は ./baselines/remi/resultに書き込まれる

# プレイさせる

## DDPG
```
python -m baselines.torcs_ddpg.play --checkpoint=/path/to/.../torcs-ddpg-XXXX-XX-XX-XX-XX-XXX-XXXXXX/model_data/epoch_XXXX.ckpt
```

## GAIL
```
python -m baselines.gail.play --load_model_path=baselines/gail/.../checkpoint/torcs_gail/torcs_gail_XXXX
```

## Hybrid (Remi)
```
python -m baselines.remi.play --load_model_path=baselines/remi/result/.../checkpoint/torcs_remi/torcs_remi_XXXX
```
