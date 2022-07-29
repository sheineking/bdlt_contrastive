# Contrastive Training
The scripts and classes in this folder are used to pretrain the embedding space of a model using contrastive learning.
Once a model was trained, its weights are frozen and a classifier is trained on top of the model's embeddings to determine if a pair of sentence constitutes a paraphrase (label=1) or not (label=0).


## Files
| File                | Description                                             |
|---------------------|---------------------------------------------------------|
| learning_manager.py | Learning Manager class to define the training process   |
| losses.py           | Available losses as classes (pairwise, triplet, infoNCE |
| model\_configs.py   | Script to write the model\_configs.json                 |
| models.py           | Model definition and optimizer selection                |



