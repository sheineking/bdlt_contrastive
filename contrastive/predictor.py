import os
import torch as T
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import models as m

WEIGHT_PATH = os.path.abspath("./models/weights/")
CUTOFF_VALS = {"Pairwise_SGD": 0.966584384441376, "Pairwise_RMS": 0.968499064445496, "Pairwise_LARS": 0.968992710113525,
               "Triplet_SGD": 0.633633017539978, "Triplet_RMS": 0.68914920091629, "Triplet_LARS": 0.658703505992889,
               "InfoNCE_SGD": 0.629370331764221, "InfoNCE_RMS": 0.597379446029663, "InfoNCE_LARS": 0.617157995700836,
               "Baseline": 0.695563197135925}

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Predictor():
    def __init__(self, model_name="Triplet_LARS"):
        """
        Function that loads the trained model and the tokenizer.
        It also sets a precalculated cutoff-value at which to classify a pair as paraphrases.

        :param model_name:  The name of the model to be loaded
        """

        self.model = m.PredictionModel()
        weight_path = WEIGHT_PATH + "/" + model_name + ".pt"

        try:
            self.cutoff = CUTOFF_VALS[model_name]
        except:
            print(f"{model_name} is not a valid model name. Please use one of the following names:")
            print(", ".join(list(CUTOFF_VALS.keys())))
            exit(1)

        print("\n" + "=" * 50)
        print(f"       Prediction: {model_name} ")
        print("=" * 50)
        print("Preparing the model...")
        try:
            model_weights = T.load(weight_path)
            self.model.load_state_dict(model_weights)

        except:
            print(f"An error occured while trying to load the weights from {weight_path}. "
                  f"Please verify the directory as well as the model architecture.")
            exit(1)

        print("Preparing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(m.MODEL_NAME)


    def tokenize_dataset(self, dataset):
        # Determine removal columns (index and all sentences)
        remove_columns = ["idx", "sentence1", "sentence2"]

        # Apply tokenization and remove unnecessary columns
        tokenized_ds = dataset.map(lambda example: self.tokenize_function(example),
                                   remove_columns=remove_columns)

        tokenized_ds = tokenized_ds.rename_column("label", "labels")

        tokenized_ds = tokenized_ds.with_format("torch")
        # Unsqueeze the label column
        self.tokenized_ds = tokenized_ds.map(lambda example: {"labels": T.unsqueeze(example["labels"], dim=0)},
                                             remove_columns=["labels"])


    def tokenize_function(self, example):
        result_dict = {}

        # Get the tokens for both sentences and add them to the result_dict
        result_dict["input1"] = self.tokenizer(example["sentence1"], padding="max_length", truncation=True)
        result_dict["input2"] = self.tokenizer(example["sentence2"], padding="max_length", truncation=True)

        # Return the tokenized sentences as a dictionary
        return result_dict


    def predict(self, return_logits=False, batch_size=32):
        """
        Function to predict on the dataset. Per default the logits will be split at a cutoff value.

        :param return_logits:   True: Return the predictions as logits
        :return:                Predictions and labels as lists
        """

        # Send the model to the device and set it into eval mode
        model = self.model.to(device)
        model.eval()

        # Prepare the dataloader (The collate_fn automatically puts each batch on the device)
        dataloader = DataLoader(self.tokenized_ds.with_format("torch", device=device), batch_size=batch_size)

        # Initialize two lists for predictions and labels
        predictions = []
        labels = []

        num_batches = len(dataloader)
        i = 0
        print("\nPredicting...")
        for batch in dataloader:
            if i % 50 == 0:
                print(f"- Processing batch {i}/{num_batches}")

            # Predictions are made based on the cutoff value
            logits = model(batch=batch)
            if not return_logits:
                logits = (logits > self.cutoff).float()

            # Append the predictions and labels for the current batch
            predictions.extend(T.squeeze(logits, dim=-1).detach().cpu().numpy().tolist())
            labels.extend(T.squeeze(batch["labels"], dim=-1).detach().cpu().numpy().tolist())

            i += 1

        return predictions, labels