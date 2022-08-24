import os
import sys

import importlib

from datasets import load_dataset, concatenate_datasets
import copy
import torch as T

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, f1_score
import numpy as np
import pandas as pd

# ===================================================================
# Preparation
# ===================================================================
MODELS = {"supervised": ["Supervised_SGD", "Supervised_RMS", "Supervised_LARS"],
          "pretrained": ["Pretrained_Pairwise", "Pretrained_Triplet", "Pretrained_InfoNCE"],
          "contrastive": ["Pairwise_SGD", "Pairwise_RMS", "Pairwise_LARS",
                          "Triplet_SGD", "Triplet_RMS", "Triplet_LARS",
                          "InfoNCE_SGD", "InfoNCE_LARS", "InfoNCE_RMS"]}

MAIN_DIR = os.path.abspath("./")

def switch_directory(folder_name='../contrastive/'):
    abs_path = os.path.abspath(folder_name + "predictor.py")
    dir_name = os.path.dirname(abs_path)

    sys.path.append(dir_name)
    os.chdir(dir_name)

# ==================================================================
# Data loading functions
# ==================================================================
def load_val_data():
    dataset = load_dataset("ContrastivePretrainingProject/contrastive_paraphrases", use_auth_token=True)

    remove_cols = ["sentence" + str(idx) for idx in range(3, 7)]

    # Add copy of dataset which moves sentence3 to sentence2,
    # so that negative and positive samples are used.
    # Additionally, add lable column
    dataset = dataset.map(lambda example: {"label": 1})
    dataset_negatives = copy.deepcopy(dataset)
    # remove paraphrases
    dataset_negatives = dataset_negatives.remove_columns(["sentence2"])
    # move negative to position of paraphrase and change label
    dataset_negatives = dataset_negatives.rename_column("sentence3", "sentence2")
    dataset_negatives = dataset_negatives.map(lambda example: {"label": 0})
    dataset_negatives = dataset_negatives.remove_columns(remove_cols[1:])
    dataset = dataset.remove_columns(remove_cols)

    for split in ['train', 'validation', 'test']:
        # No shuffling to have the same labels for all
        dataset[split] = concatenate_datasets([dataset[split], dataset_negatives[split]])

    return dataset["validation"]


def load_glue_dataset():
    return load_dataset(path="glue", name="mrpc")["validation"]


# ===================================================================
# Main function
# ===================================================================
def perform_prediction(dataset, folder_name="supervised", batch_size=4, return_logits=False):
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

    switch_directory("../" + folder_name + "/")
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

    os.chdir(MAIN_DIR)


def find_optimal_cutoffs(logits_path="./logits.csv", out_path="./cutoff_values.csv", target="f1"):
    df = pd.read_csv(logits_path)

    # Drop all unnecessary columns (unnamed; [1:] drops "labels")
    columns = df.keys().values.tolist()
    model_names = [name for name in columns if not "Unnamed" in name][1:]
    skip_models = []

    optimizer_func = find_cutoff_f1 if target == "f1" else find_cutoff_roc

    if not os.path.isfile(out_path):
        with open(out_path, "w") as f:
            f.write("model_name,cutoff\n")
    else:
        existing_df = pd.read_csv(out_path)
        skip_models = list(existing_df["model_name"])

    for model in model_names:
        if model in skip_models:
            print(f"{model} was already recorded. This model will be skipped.")
            continue

        cutoff_value = optimizer_func(df["labels"], df[model])
        with open(out_path, "a") as f:
            f.write(f"{model},{cutoff_value}\n")


def find_cutoff_roc(labels, logits):
    """
    Find the optimal probability cutoff point for a model based on AUROC.

    :param labels:      The correct labels
    :param logits:      The logits calculated by each model
    :return:            The optimal cutoff value
    """
    fpr, tpr, thresholds = roc_curve(labels, logits)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t['threshold'].values[0]

def find_cutoff_f1(labels, logits):
    """
    Find the optimal probability cutoff point for a model based on precision and recall.

    :param labels:      The correct labels
    :param logits:      The logits calculated by each model
    :return:            The optimal cutoff value
    """
    precision, recall, thresholds = precision_recall_curve(labels, logits)
    # Drop the last elements as these are 1 and 0 respectively (and thresholds contains one less element)
    precision = precision[:-1].copy()
    recall = recall[:-1].copy()

    i = np.arange(len(precision))
    f1 = pd.DataFrame({'val': pd.Series(2*precision*recall/(precision+recall+1e-10), index=i),
                       'threshold': pd.Series(thresholds, index=i)})
    f1_t = f1.iloc[f1.val.argsort()[-1:]]

    return f1_t['threshold'].values[0]



def get_f1_and_conf_from_predictions(predictions_path="./predictions.csv", out_path="./f1_scores.csv"):
    prediction_df = pd.read_csv(predictions_path)
    get_f1_and_conf(prediction_df=prediction_df, out_path=out_path)


def get_f1_and_conf_from_logits(logits_path="./logits.csv", cutoff_path="./cutoff_values.csv",
                                out_path="./f1_scores.csv"):

    df = pd.read_csv(logits_path)
    cutoff_df = pd.read_csv(cutoff_path)
    cutoff_dict = dict(zip(cutoff_df["model_name"], cutoff_df["cutoff"]))

    prediction_df = pd.DataFrame()
    prediction_df["labels"] = df["labels"]

    # Get the predictions for each model
    for model_name, cutoff in cutoff_dict.items():
        prediction_df[model_name] = 0
        prediction_df.loc[df[model_name] > cutoff, model_name] = 1

    get_f1_and_conf(prediction_df=prediction_df, out_path=out_path)




def get_f1_and_conf(prediction_df: pd.DataFrame(), out_path=".f1_scores.csv"):
    name_list = []
    f1_list = []

    # Drop all unnecessary columns (unnamed; [1:] drops "labels")
    columns = prediction_df.keys().values.tolist()
    model_names = [name for name in columns if not "Unnamed" in name][1:]

    # Get the f1-scores and print them with the confusion matrix
    for model_name in model_names:
        print(model_name)
        print(confusion_matrix(prediction_df['labels'], prediction_df[model_name]))
        f1 = f1_score(prediction_df['labels'], prediction_df[model_name])
        print(f1)
        print("-" * 50 + "\n\n")

        name_list.append(model_name)
        f1_list.append(f1)

    # Save the f1 scores
    f1_df = pd.DataFrame(list(zip(name_list, f1_list)), columns=["model_name", "f1"])
    f1_df.to_csv(out_path)