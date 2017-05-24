# FMA_convnet_features

Keunwoo Choi, May 2017

A repo to host convnet features for FMA.

## What is convnet feature?

A set of features that is computed from [transfer learning repo](https://github.com/keunwoochoi/transfer_learning_music)

## What is FMA?

[Free Music Archieve dataset](https://github.com/mdeff/fma)

## How good is it?
A bit better than the provided audio features in FMA -- showed 63.94% of accuracy. +1% improvement over baseline. Although I think this convnet feature also should be considered as one of baseline features.

### Results (Compare these with the results from [provided baselines](https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb))

![result](https://github.com/keunwoochoi/FMA_convnet_features/blob/master/results_table.png)

### T-SNE

![tsne on small](https://github.com/keunwoochoi/FMA_convnet_features/blob/master/tsne_feature_small.png)

![tsne on medium](https://github.com/keunwoochoi/FMA_convnet_features/blob/master/tsne_feature_medium.png)

## Details

* This is computed from FMA `large`, which is 30-second previews clips

