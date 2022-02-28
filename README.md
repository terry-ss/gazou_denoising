# 画像のノイズ除去

このコートの要点は、パディングで全て画像が同じサイズになるや、プーリング層を除去や、大きいエポック数など。

## 環境

python=3.9.7

pytorch==1.10.0

pytorch-lightning==1.5.7

## データ準備

データリンク：https://www.kaggle.com/c/denoising-dirty-documents/data
解凍されて、以下通り置きます。
````
├── gazou_denosing/
│   ├── data/
│       ├── train/
│       ├── train_clean/
│       ├── test/
│       ├── sampleSubmission.csv

````

データの反転は効果を少し上げることができる。
```bash
python　data_aug.py
```

訓練とバリデーションのファイルの情報はデータベースに記入します。
```bash
python　train_val_dataset.py
```


## フィット

###  訓練
```bash
python train.py 
```

###  サブミット
```bash
mkdir data/test_cleaned/
python　submission.py
```
そして、v_numを端末で入力します。


## 参考
https://github.com/toshi-k/Kaggle-Denoising-Dirty-Documents
