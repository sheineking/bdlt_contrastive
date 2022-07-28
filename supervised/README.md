# Supervised Training
The structure of this folder mimics that of [contrastive](../contrastive). Here, however, the embedding model is trained together with the classifier instead of the former being pre-trained in a contrastive manner.
Equally, the classifier is trained to determine if a pair of sentence constitutes a paraphrase (label=1) or not (label=0).

## Usage
In order to train a model, use the following command. The config flag is used to determine the baseline configuration (refer to ./models/model_configs.json for all available configurations).

```
python main.py --config=Supervised_SGD
```

If parameters from the baseline configuration should be adapted, simply pass their name as a flag. 
For instance, if the epochs should be set to 15 but everything else should remain the same, use the following command:

```
python main.py --config=Supervised_SGD --epochs=15
```

## Files
| File                | Description                                             |
|---------------------|---------------------------------------------------------|
| main.py             | Main script used to interact with the other files       |
| learning_manager.py | Learning Manager class to define the training process   |
| model\_configs.py   | Script to write the model\_configs.json                 |
| models.py           | Model definition and optimizer selection                |



