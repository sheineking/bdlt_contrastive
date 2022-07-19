import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import torch as T
from torch.utils.data import DataLoader

import models as m
import contrastive_losses as l

DATASETS = {"Test_Dataset": {"path": "glue",
                             "name": "mrpc"}
            }
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQ_LEN = 128

# Source: https://jamesmccaffrey.wordpress.com/2022/03/17/yet-another-siamese-neural-network-example-using-pytorch/


# ================================================================
# Preparation
# ================================================================
device = T.device("cuda" if T.cuda.is_available() else "cpu")


# ================================================================
# Main class
# ================================================================
class LearningManager():
    def __init__(self,  model_name=MODEL_NAME, loss_funct="pairwise"):
        print("\n" + "="*50)
        print(f"    {model_name}")
        print("="*50)

        print("Preparing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Preparing the model...")
        if loss_funct == "pairwise":
            self.model = m.PairwiseContrastive()
            self.loss = l.PairwiseLoss()



    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    def load_dataset(self, dataset_name="Test_Dataset"):
        """
        Function to load the model specified by dataset_name. Sets the dataset-attribute
        :param dataset_name:    Key in the DATASETS-dictionary
        """
        self.dataset = load_dataset(**DATASETS[dataset_name])



    # Todo: Adapt this to our dataset
    # Source: https://huggingface.co/docs/transformers/training
    def tokenize_data(self):
        """
        Function to tokenize the attribute dataset and split it into train_ds and eval_ds for training.
        """

        if not hasattr(self, "dataset"):
            self.load_dataset()

        # Apply tokenization
        tokenized_ds = self.dataset.map(lambda example: self.tokenize_function(example),
                                        remove_columns=["idx", "sentence1", "sentence2"])

        # Preprocess the tokenized dataset
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.with_format("torch", device=device)

        # Assign the splits
        self.train_ds = tokenized_ds["train"]
        self.eval_ds = tokenized_ds["validation"]
        self.test_ds = tokenized_ds["test"]



    def tokenize_function(self, example):
        # Tokenize both sentences
        tokens_dict1 = self.tokenizer(example["sentence1"], padding="max_length", truncation=True,
                                      max_length=MAX_SEQ_LEN)

        tokens_dict2 = self.tokenizer(example["sentence2"], padding="max_length", truncation=True,
                                      max_length=MAX_SEQ_LEN)

        # Rename the keys of the two token dictionaries to include everything on one level
        #tokens_dict1["input_ids1"] = tokens_dict1.pop("input_ids")
        #tokens_dict1["attention_mask1"] = tokens_dict1.pop("attention_mask")
        #tokens_dict1["token_type_ids1"] = tokens_dict1.pop("token_type_ids")

        #tokens_dict2["input_ids2"] = tokens_dict2.pop("input_ids")
        #tokens_dict2["attention_mask2"] = tokens_dict2.pop("attention_mask")
        #tokens_dict2["token_type_ids2"] = tokens_dict2.pop("token_type_ids")

        # Return them as a dictionary
        return {"input1": tokens_dict1, "input2": tokens_dict2}   #{**tokens_dict1, **tokens_dict2}




    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def conduct_training(self, epochs=10, lr=0.005, optimizer='sgd', batch_size=2, subset=None):
        """
        Function that performs training on the train_ds and validates on the eval_ds.

        :param epochs:          How many epochs to train
        :param lr:              The learning rate
        :param optimizer:       The optimizer
        :param batch_size:      Batch size used in training
        :param subset:          Optional: If only a subset of the data should be used
        :return:
        """

        if not hasattr(self, "train_ds"):
            self.tokenize_data()

        model = self.model.to(device)

        optimizer = T.optim.SGD(model.parameters(), lr=lr)

        # Prepare the two dataloaders
        train_data = self.train_ds.select(range(subset)) if subset is not None else self.train_ds
        eval_data = self.eval_ds.select(range(subset)) if subset is not None else self.eval_ds

        train_dl = DataLoader(train_data, batch_size=batch_size)
        eval_dl = DataLoader(eval_data, batch_size=batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            with T.enable_grad():
                # Set the model into train mode
                model.train()
                train_loss = self.loss_epoch(model=model, dataloader=train_dl, optimizer=optimizer)

            # Perform evaluation
            model.eval()
            with T.no_grad():
                val_loss = self.loss_epoch(model=model, dataloader=eval_dl)

            # Print an update
            print("Train-Loss = %10.4f  |   Validation-Loss = %10.4f\n" %(train_loss, val_loss))


            # Todo: Perform checkpointing




    def loss_batch(self, emb1, emb2, label, optimizer=None):
        """
        Function to calculate the loss on one batch
        :param emb1:            The embeddings for the first sentences
        :param emb2:            The embeddings for the second sentences
        :param label:           The labels for each of the pairs in the batch
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the batch
        """
        loss = self.loss(emb1, emb2, label)

        if optimizer is not None:
            # Reset the gradients, compute new ones and perform a weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def loss_epoch(self, model, dataloader, optimizer=None):
        """
        Function to calculate the loss for epoch
        :param model:           The model used in the epoch
        :param dataloader:      The dataloader to obtain batched data
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the entire epoch (normalized by the number of datapoints)
        """


        # Reset the loss at the beginning of each epoch
        ep_loss = 0.0
        ds_len = len(dataloader.dataset)

        # Loop over all batches in the data
        for batch in dataloader:
            # Obtain the necessary inputs
            X1 = batch['input1']
            X2 = batch['input2']
            label = batch['labels']

            # Get the embeddings
            emb1, emb2 = model(X1, X2)

            # Compute the loss value
            loss_val = self.loss_batch(emb1=emb1, emb2=emb2, label=label, optimizer=optimizer)

            # Update the running loss
            ep_loss += loss_val

        # Return the normalized loss
        return ep_loss / ds_len










        # Compute new gradients
        # loss_val.required_grad = True
        loss_val.backward()

        # Update the weights
        optimizer.step()

        return loss_val

