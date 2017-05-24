# FMA_convnet_features

Keunwoo Choi, May 2017

A repo to host convnet features for FMA. Wanted to do more experiment than baseline method, but don't think I have time to do so, so just releasing it.

### What is convnet feature?

A set of features that is computed from [transfer learning repo](https://github.com/keunwoochoi/transfer_learning_music)

### What is FMA?

[Free Music Archieve dataset](https://github.com/mdeff/fma): A large collection of audio and its genre annotation released in 2017.

### How good is the feature?
 * In general, it achieved better performance than some popular audio features [as in this paper](https://arxiv.org/abs/1703.09179)
 * For FMA dataset, it's bit better than the provided audio features -- showed 63.94% of accuracy. +1% improvement over baseline. Although I think this convnet feature also should be considered as one of baseline features.

### How to use?

```python
# Load the features
import numpy as np
feat1 = np.load('fma_large_layer1.npy')
feat2 = np.load('fma_large_layer2.npy')
feat3 = np.load('fma_large_layer3.npy')
feat4 = np.load('fma_large_layer4.npy')
feat5 = np.load('fma_large_layer5.npy')

# concatenate the features
features = np.concatenate((feat1, feat2, feat3, feat4, feat5), axis=1)
features.shape
# (106574, 160)
# This is matched to the order of metadata.csv provided in FMA.
# Now use it for your task!
```

### Classification results (Compare these with the results from [provided baselines](https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb))

![result](https://github.com/keunwoochoi/FMA_convnet_features/blob/master/results_table.png)

### T-SNE
#### on FMA-'small' 

![tsne on small](https://github.com/keunwoochoi/FMA_convnet_features/blob/master/tsne_feature_small.png)

#### on FMA-'medium'
![tsne on medium](https://github.com/keunwoochoi/FMA_convnet_features/blob/master/tsne_feature_medium.png)

### Details

* This is computed from FMA `large`, which is 30-second previews clips
