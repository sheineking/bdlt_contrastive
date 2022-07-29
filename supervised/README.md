# Supervised Training
The structure of this folder mimics that of [contrastive](../contrastive). Here, however, the embedding model is trained together with the classifier instead of the former being pre-trained in a contrastive manner.
Equally, the classifier is trained to determine if a pair of sentence constitutes a paraphrase (label=1) or not (label=0).


## Files
| File                | Description                                             |
|---------------------|---------------------------------------------------------|
| learning_manager.py | Learning Manager class to define the training process   |
| model\_configs.py   | Script to write the model\_configs.json                 |
| models.py           | Model definition and optimizer selection                |



