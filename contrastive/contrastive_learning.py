from datasets import load_dataset
from transformers import AutoTokenizer
import torch as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time

import models as m
import contrastive_losses as l
import dataset_prep_dummy as d


# ================================================================
# Constants
# ================================================================

DATASETS = {"pairwise": {"path": "glue",
                         "name": "mrpc"},
            "triplet": {"path": "glue",
                        "name": "mrpc"},
            "infoNCE": {"path": "glue",
                        "name": "mrpc"}
            }

TRAIN_MODES = {"pairwise": {"model": m.ContrastiveModel(),
                            "loss": l.PairwiseLoss()},
               "triplet": {"model": m.ContrastiveModel(),
                            "loss": l.TripleLoss()},
               "infoNCE": {"model": m.ContrastiveModel(),
                            "loss": l.InfoNCE_Loss()}}

MAX_SEQ_LEN = 128
MODEL_OUT_PATH = './models/'

# ================================================================
# Preparation
# ================================================================
device = T.device("cuda" if T.cuda.is_available() else "cpu")


# ================================================================
# Main class
# ================================================================
class LearningManager():
    def __init__(self,  train_mode="pairwise", model_name=None):
        """
        Defines the instance of the LearningManager
        :param train_mode:      ["pairwise", "triple_loss", "infoNCE"]. Refers to the keys in TRAIN_MODES
        :param model_name:      Optional: Name to identify weights and logs; If None, a name is constructed
        """

        print("\n" + "="*50)
        print("       Contrastive Learning ")
        print("="*50)

        print("Preparing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(m.MODEL_NAME)

        print("Preparing the model...")
        self.train_mode = train_mode
        self.model = TRAIN_MODES[train_mode]["model"]
        self.loss = TRAIN_MODES[train_mode]["loss"]


        if model_name is None:
            time_stamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            model_name = m.MODEL_NAME.split("/")[1] + "_" + train_mode + "_" + time_stamp

        self.weight_path = MODEL_OUT_PATH + "/weights/" + model_name + ".pt"
        self.log_path = MODEL_OUT_PATH + "/logs/" + model_name + "/"

        print("Initial preparation completed.")
        print(f"- Model weights will be saved to: {self.weight_path}")
        print(f"- Training logs will be saved to {self.log_path}")

        # Ensure that the weights and logs folder exist
        self.create_model_folders()


    def create_model_folders(self):
        if not os.path.exists(MODEL_OUT_PATH):
            os.mkdir(MODEL_OUT_PATH)

        weights_path = MODEL_OUT_PATH + '/weights/'
        log_path = MODEL_OUT_PATH + '/logs/'

        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)



    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    def load_dataset(self):
        """
        Function to load the model specified by dataset_name. Sets the dataset-attribute
        :param dataset_name:    Key in the DATASETS-dictionary
        """
        self.dataset = load_dataset(**DATASETS[self.train_mode])

        # Todo: Remove for final dataset
        # Create a dummy set
        if self.train_mode == "triplet":
            self.dataset = d.build_triplet_ds(self.dataset)
        if self.train_mode == "infoNCE":
            self.dataset = d.build_infoNCE_ds(self.dataset)


        # Determine number of sentences in the dataset
        self.num_sentences = 1
        features = self.dataset["train"].features
        while True:
            try:
                _ = features['sentence' + str(self.num_sentences)]
                self.num_sentences += 1
            except:
                break
        # Subtract one from the number of sentences
        self.num_sentences = self.num_sentences - 1



    # Todo: Adapt this to our dataset
    # Source: https://huggingface.co/docs/transformers/training
    def tokenize_data(self):
        """
        Function to tokenize the attribute dataset and split it into train_ds, eval_ds and test_ds
        """

        if not hasattr(self, "dataset"):
            self.load_dataset()

        # Determine removal columns (index and all sentences)
        remove_columns = ["idx"]
        for num in range(1, self.num_sentences + 1):
            remove_columns.append("sentence" + str(num))

        # Apply tokenization and remove unnecessary columns
        tokenized_ds = self.dataset.map(lambda example: self.tokenize_function(example),
                                        remove_columns=remove_columns)

        # Preprocess the tokenized dataset and send it to the device
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.with_format("torch", device=device)

        # Assign the splits
        self.train_ds = tokenized_ds["train"]
        self.eval_ds = tokenized_ds["validation"]
        self.test_ds = tokenized_ds["test"]


    def tokenize_function(self, example):
        # Create empty dictionary
        result_dict = {}
        # Tokenize all sentences
        for num in range(1, self.num_sentences + 1):
            sentence = example["sentence" + str(num)]
            tokens_dict = self.tokenizer(sentence, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

            # Add the tokens_dict to the result_dict and increase the iterator
            result_dict["input" + str(num)] = tokens_dict

        # Return them as a dictionary
        return result_dict




    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def conduct_training(self, epochs=10, lr=0.005, optimizer_name='sgd', batch_size=2, subset=None):
        """
        Function that performs training on the train_ds and validates on the eval_ds.
        Checkpointing is performed based on validation loss.

        :param epochs:          How many epochs to train
        :param lr:              The learning rate
        :param optimizer_name:  String to identify the optimizer
        :param batch_size:      Batch size used in training
        :param subset:          Optional: If only a subset of the data should be used
        """

        if not hasattr(self, "train_ds"):
            self.tokenize_data()

        # Prepare model, optimizer and summary writer
        model = self.model.to(device)
        optimizer = T.optim.SGD(model.parameters(), lr=lr)
        writer = SummaryWriter(log_dir=self.log_path)

        # Prepare the two dataloaders
        train_data = self.train_ds.select(range(subset)) if subset is not None else self.train_ds
        eval_data = self.eval_ds.select(range(subset)) if subset is not None else self.eval_ds
        train_dl = DataLoader(train_data, batch_size=batch_size)
        eval_dl = DataLoader(eval_data, batch_size=batch_size)

        # Initialize the validation loss used for checkpointing
        best_val_loss = float('inf')

        print("\nPerforming training based on the following parameters:")
        print(f"- Epochs:           {epochs}")
        print(f"- Batchsize:        {batch_size}")
        print(f"- Learning Rate:    {lr}")
        print(f"- Num sentences:    {self.num_sentences}")
        print(f"- Optimizer:        {optimizer}")
        print(f"- Loss:             {self.loss}")

        for epoch in range(epochs):
            print("\n" + "-" * 100)
            print(f"Epoch {epoch+1}/{epochs}")

            with T.enable_grad():
                # Set the model into train mode
                model.train()
                train_loss = self.loss_epoch(model=model, dataloader=train_dl, optimizer=optimizer)

            # Perform evaluation
            model.eval()
            with T.no_grad():
                val_loss = self.loss_epoch(model=model, dataloader=eval_dl)

            # Perform checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                T.save(model.state_dict(), self.weight_path)
                print(f"New checkpoint for validation loss. Model weights saved to {self.weight_path}")

            # Write the logs
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            # Print an update
            print("Train-Loss = %10.8f  |   Validation-Loss = %10.8f" % (train_loss, val_loss))


        # Close the writer
        writer.flush()
        writer.close()





    def loss_epoch(self, model, dataloader, optimizer=None):
        """
        Function to calculate the loss for epoch
        :param model:           The model used in the epoch
        :param dataloader:      The dataloader to obtain batched data
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the entire epoch (normalized by the number of data points)
        """


        # Reset the loss at the beginning of each epoch
        ep_loss = 0.0
        ds_len = len(dataloader.dataset)

        # Loop over all batches in the data
        for batch in dataloader:
            # Obtain the necessary inputs
            label = batch['labels']

            # Get the embeddings as a dictionary
            embeddings = model(batch=batch, num_sentences=self.num_sentences)

            # Compute the loss value based on the label and embeddings; Optimizer is passed in case of usage with train
            loss_val = self.loss_batch(embeddings=embeddings, label=label, optimizer=optimizer)

            # Update the running loss
            ep_loss += loss_val

        # Return the normalized loss
        return ep_loss / ds_len


    def loss_batch(self, embeddings, label, optimizer=None):
        """
        Function to calculate the loss on one batch
        :param embeddings:      The embeddings of the current batch in a dictionary
        :param label:           Optional: In case of pairwise loss, this contains the labels for each of the pairs
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the batch
        """
        loss = self.loss(embeddings=embeddings, label=label, num_sentences=self.num_sentences)

        if optimizer is not None:
            # Reset the gradients, compute new ones and perform a weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()



