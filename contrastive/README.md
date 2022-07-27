# Contrastive Training
The scripts and classes in this folder are used to pretrain the embedding space of a model using contrastive learning.
Once a model was trained, its weights are frozen and a classifier is trained on top of the model's embeddings to determine if a pair of sentence constitutes a paraphrase (label=1) or not (label=0).

## Usage
In order to pretrain a model, use the following command. The config flag is used to determine the baseline configuration (refer to ./models/model\_configs.json for all available configurations).

```
python contrastive.py --config=Pairwise_LARS
```

If parameters from the baseline configuration should be adapted, simply pass their name as a flag. 
For instance, if the epochs should be set to 15 but everything else should remain the same, use the following command:

```
python contrastive.py --config=Pairwise_LARS --epochs=15
```

## Files
| File                     | Description                                             |
|--------------------------|---------------------------------------------------------|
| contrastive.py           | Main script used to interact with the other files       |
| contrastive\_learning.py | Learning Manager class to define the training process   |
| contrastive\_losses.py   | Available losses as classes (pairwise, triplet, infoNCE |
| json\_config.py          | Script to write the model\_configs.json                 |
| models.py                | Model definition and optimizer selection                |



