import os
import sys
import pandas as pd

from datasets import load_dataset
import torch as T

from sklearn.metrics import roc_curve, auc
import numpy as np

# ===================================================================
# Preparation
# ===================================================================
DATASET = {"path": "glue", "name": "mrpc"}

MODELS = {"supervised": ["Supervised_SGD", "Supervised_RMS", "Supervised_LARS"],
          "pretrained": ["Pretrained_Pairwise", "Pretrained_Triplet", "Pretrained_InfoNCE"],
          "contrastive": ["Pairwise_SGD", "Pairwise_RMS", "Pairwise_LARS",
                          "Triplet_SGD", "Triplet_LARS", "Triplet_LARS",
                          "InfoNCE_SGD", "InfoNCE_RMS", "InfoNCE_LARS"]}

MAIN_DIR = os.path.abspath("./")

def switch_directory(folder_name='./contrastive/'):
    abs_path = os.path.abspath(folder_name + "predictor.py")
    dir_name = os.path.dirname(abs_path)

    sys.path.append(dir_name)
    os.chdir(dir_name)


# ===================================================================
# Main function
# ===================================================================
def perform_prediction(folders=None, batch_size=4, return_logits=False):
    if folders is None:
        folders = ["supervised", "pretrained", "contrastive"]

    dataset = load_dataset(**DATASET)["validation"]

    out_path = os.path.abspath("./logits.csv" if return_logits else "./predictions.csv")

    # Check if a CSV-file already exists and if so, load its columns to skip these models
    df = pd.DataFrame()
    i = 0
    processed_models = []
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
        processed_models = list(df.columns)[2:]
        i = len(processed_models)
        print(f"An existing csv was found under {out_path}.")
        print(f"The following models will be skipped: {processed_models}")



    for folder_name in folders:
        switch_directory("./" + folder_name + "/")
        model_names = MODELS[folder_name]
        import predictor as p

        for model_name in model_names:
            if model_name in processed_models:
                print(f"Predictions for {model_name} were already recorded. This model will be skipped.")
                continue

            Predictor = p.Predictor(model_name=model_name)
            Predictor.tokenize_dataset(dataset)

            predictions, labels = Predictor.predict(return_logits=return_logits, batch_size=batch_size)

            # If the first iteration is run, save the labels (they will stay the same afterwards)
            if i == 0:
                df["labels"] = labels
            df[model_name] = predictions

            # Save after each round as a safeguard
            df.to_csv(out_path)
            print(f"Saved results to {out_path}")

            i += 1
            T.cuda.empty_cache()

        # Get back to the directory of the script
        os.chdir(MAIN_DIR)



def find_optimal_cutoffs(logits_path="./logits.csv", out_path="./cutoff_values.csv"):
    df = pd.read_csv(logits_path)
    model_names = list(df.columns)[2:]

    if not os.path.isfile(out_path):
        with open(out_path, "w") as f:
            f.write("model_name,cutoff\n")

    for model in model_names:
        cutoff_value = find_single_cutoff(df["labels"], df[model])
        with open(out_path, "a") as f:
            f.write(f"{model},{cutoff_value}\n")






def find_single_cutoff(labels, logits):
    """
    Find the optimal probability cutoff point for a model based on the logits.

    :param labels:      The correct labels
    :param logits:      The logits calculated by each model
    :return:            The optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(labels, logits)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t['threshold'].values[0]
