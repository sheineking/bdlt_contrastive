# Supervised Training
The structure of this folder mimics that of [contrastive](../contrastive). Here, however, supervised learning is performed to train a classifier to determine if a pair of sentence constitutes a paraphrase (label=1) or not (label=0).

This can be done in two ways:
1. Benchmark model
To train a benchmark model, simply call the main script with mode==supervised:

```
python main.py --mode=supervised --config=Supervised_SGD
```

2. Classifier on top of pre-trained model
In order to train a classifier on top of a model that was pre-trained in a contrastive fashion, simply provide the name of the contrastive model under the encoder flag:
```
python main.py --mode=supervised --config=Supervised_SGD --encoder=Pairwise_SGD
```
The weights of the encoder will be frozen and only the classifier trained on top.


## Files
| File                | Description                                             |
|---------------------|---------------------------------------------------------|
| learning_manager.py | Learning Manager class to define the training process   |
| model\_configs.py   | Script to write the model\_configs.json                 |
| models.py           | Model definition and optimizer selection                |



