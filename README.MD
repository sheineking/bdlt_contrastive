# Paraphrase detection with contrastive pretraining
The goal of this repository is to evaluate the effect of contrastive pretraining on classification performance 
in paraphrase detection. The folder [contrastive](./contrastive) contains all the code to pretrain an encoder and later fit a classifier 
on top.
The folder [supervised](./supervised) follows as similar structure and is used to train the encoder and classifier at once.
This is meant to create a baseline model against which to compare the performance of the contrastive model.

## Setup
In order to use the repository, install all packages from requirements.txt
```
pip install -r 'requirements.txt'
```


## Training
The training (contrastive and supervised) can be done using the [main.py](./main.py) script by passing the required flags.
- The mode-flag is used to switch between contrastive and supervised
- The config flag determines the baseline configuration 
  - Refer to model_configs.json for all available configurations. They can be found in the models folder of contrastive and supervised   

Example
```
python main.py --mode=contrastive --config=Pairwise_LARS
```

If parameters from the baseline configuration should be adapted, simply pass their name as a flag. 
For instance, if the epochs should be set to 15 but everything else should remain the same, use the following command:

```
python main.py --mode=contrastive --config=Pairwise_LARS --epochs=15
```


## Sweeping
In order to obtain optimal hyperparameters, the script in [sweeping](./sweeping) runs sweeps with [Weights and Biases](www.wandb.ai).
The usage is similar to the main.py script. Inside the sweeping-folder execute the following command:
```
python sweeping.py --mode=contrastive
```
