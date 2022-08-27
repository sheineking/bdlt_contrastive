# Paraphrase detection with contrastive pre-training
The goal of this repository is to evaluate the effect of contrastive pre-training on classification performance 
in paraphrase detection. The folder [contrastive](./contrastive) contains all the code to pretrain an encoder.
The folder [supervised](./supervised) follows as similar structure and is used to train the encoder and classifier at once.
This is meant to create a baseline model against which to compare the performance of the contrastive model.
Finally, [pretrained](./pretrained) is used to train a linear layer on top of the [contrastive](./contrastive) models after freezing the encoder weights.

## Setup
In order to use the repository, install all packages from requirements.txt
```
pip install -r 'requirements.txt'
```


## Training
The training (contrastive, supervised, and pretrained) can be done using the [main.py](./main.py) script by passing the required flags.
- The mode-flag is used to switch between contrastive, supervised, and pretrained
- The config flag determines the baseline configuration 
  - Refer to the respective model_configs.json for all available configurations. They can be found in the models folder, e.g. [contrastive/models/model_configs.json]([contrastive/models/model_configs.json])

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

## Model cards
The model cards are part of the README in each folder and give an overview on the respective models. 

## Evaluation
The F1-scores, precision and recall values for each model can be found in the [evaluation](./evaluation) folder.
The columns relate to the follow datasets which are available on request:
- Val = [Custom validation dataset](https://huggingface.co/datasets/ContrastivePretrainingProject/contrastive_paraphrases)
- Test = [Custom test dataset](https://huggingface.co/datasets/ContrastivePretrainingProject/contrastive_paraphrases)
- noObf = [No obfuscation subset of PAN-13](https://huggingface.co/datasets/ContrastivePretrainingProject/pan_evaluation)
- randomObf = [Random obfuscation subset of PAN-13](https://huggingface.co/datasets/ContrastivePretrainingProject/pan_evaluation)
- translationObf = [Translation obfuscation subset of PAN-13](https://huggingface.co/datasets/ContrastivePretrainingProject/pan_evaluation)