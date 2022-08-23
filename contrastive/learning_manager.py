import copy

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
import torch as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

import os
import time

import models as m
import losses as l
import dataset_prep_dummy as d
import pandas as pd


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

MODEL_OUT_PATH = os.path.abspath('./models/')

# ================================================================
# Preparation
# ================================================================
device = T.device("cuda" if T.cuda.is_available() else "cpu")


# ================================================================
# Main class
# ================================================================
class LearningManager():
    def __init__(self,  train_mode="pairwise", model_name=None, encoder=None, use_wandb=False):
        """
        Defines the instance of the LearningManager
        :param train_mode:      ["pairwise", "triple_loss", "infoNCE"]. Refers to the keys in TRAIN_MODES
        :param model_name:      Optional: Name to identify weights and logs; If None, a name is constructed
        :param encoder:         No effect; Used for compatibility with the supervised modul
        :param use_wandb:       True: Training is conducted as part of wandb sweeping
        """

        print("\n" + "="*50)
        print("       Contrastive Learning ")
        print("="*50)

        print("Preparing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(m.MODEL_NAME)

        print("Preparing the model...")
        self.train_mode = train_mode
        # Create a deep copy to ensure that training starts fresh
        self.model = copy.deepcopy(TRAIN_MODES[train_mode]["model"])
        self.loss = TRAIN_MODES[train_mode]["loss"]


        if model_name is None:
            time_stamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            model_name = m.MODEL_NAME.split("/")[1] + "_" + train_mode + "_" + time_stamp

        self.weight_path = MODEL_OUT_PATH + "/weights/" + model_name + ".pt"
        self.log_path = MODEL_OUT_PATH + "/tensorboard_logs/" + model_name + "/"
        self.csv_path = MODEL_OUT_PATH + "/csv_logs/" + model_name + ".csv"

        print("Initial preparation completed.")

        if not use_wandb:
            print(f"- Model weights will be saved to: {self.weight_path}")
            print(f"- Tensorboard logs will be saved to: {self.log_path}")
            print(f"- CSV logs will be saved to: {self.csv_path}\n\n")

            # Ensure that the weights and logs folder exist
            self.create_model_folders()
        self.use_wandb = use_wandb


    def create_model_folders(self):
        if not os.path.exists(MODEL_OUT_PATH):
            os.mkdir(MODEL_OUT_PATH)

        subfolder_list = ['/weights/', '/tensorboard_logs/', '/csv_logs/']
        for folder in subfolder_list:
            path = MODEL_OUT_PATH + folder

            if not os.path.exists(path):
                os.mkdir(path)




    def load_dataset(self):
        """
        Function to load an already preprocessed dataset from csv.
        Format should be:
        anchor, paraphrase, neg1, neg2, neg3...
        """

        # Load the glue dataset if used in sweeping
        if self.use_wandb:
            self.load_dataset_glue()

        else:
            self.dataset = load_dataset("ContrastivePretrainingProject/contrastive_paraphrases", use_auth_token=True)


            if self.train_mode == "pairwise":
                remove_cols = ["sentence" + str(idx) for idx in range(3, 7)]
                self.num_sentences = 2
                #
                # Add copy of dataset which moves sentence3 to sentence2,
                # so that negative and positive samples are used.
                # Additionally, add lable column
                self.dataset = self.dataset.map(lambda example: {"label": 1})
                dataset_negatives = copy.deepcopy(self.dataset)
                # remove paraphrases
                dataset_negatives = dataset_negatives.remove_columns(["sentence2"])
                # move negative to position of paraphrase and change label
                dataset_negatives = dataset_negatives.rename_column("sentence3", "sentence2")
                dataset_negatives =dataset_negatives.map(lambda example: {"label": 0})

                dataset_negatives = dataset_negatives.remove_columns(remove_cols[1:])
                self.dataset = self.dataset.remove_columns(remove_cols)

                print(self.dataset)
                print(dataset_negatives)
                for split in ['train', 'validation', 'test']:

                    self.dataset[split] = concatenate_datasets([self.dataset[split], dataset_negatives[split]]).shuffle(seed=42)
                print(self.dataset)

            # Create a set of negative
            if self.train_mode == "triplet":
                remove_cols = ["sentence" + str(idx) for idx in range(4, 7)]
                self.num_sentences = 3
                self.dataset = self.dataset.remove_columns(remove_cols)


            if self.train_mode == "infoNCE":
                self.num_sentences = 6

            # make sure all pairs contain enough samples
            self.dataset = self.dataset.filter(lambda example: example["sentence" + str(self.num_sentences)] != "")
            self.dataset = self.dataset.filter(lambda example: example["sentence" + str(self.num_sentences)] != None)


    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    def load_dataset_glue(self):
        """
        Function to load the model specified by dataset_name. Sets the dataset-attribute
        :param dataset_name:    Key in the DATASETS-dictionary
        """
        self.dataset = load_dataset(**DATASETS[self.train_mode])

        if self.train_mode != "pairwise":
            HardNegativePreparer = d.HardNegativePreparer()

            # Create a set of negative
            if self.train_mode == "triplet":
                self.dataset = HardNegativePreparer.build_dataset_with_negatives(dataset=self.dataset, n=1)
            if self.train_mode == "infoNCE":
                self.dataset = HardNegativePreparer.build_dataset_with_negatives(dataset=self.dataset, n=4)


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

        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.with_format("torch")

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

            # Pad to the maximum length of the model
            tokens_dict = self.tokenizer(sentence, padding="max_length", truncation=True)

            # Add the tokens_dict to the result_dict and increase the iterator
            result_dict["input" + str(num)] = tokens_dict

        if not "label" in example:
            result_dict["label"] = 1
        # Return them as a dictionary
        return result_dict




    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def conduct_training(self, epochs=15, batch_size=16, optimizer_name='sgd', lr=0.0055, momentum=0.7, weight_decay=0,
                         alpha=0.99, eps=1e-08, trust_coef=0.001, stopping_patience=3, subset=None):
        """
        Function that performs training on the train_ds and validates on the eval_ds.
        Checkpointing is performed based on validation loss.

        :param epochs:              How many epochs to train
        :param batch_size:          Batch size used in training

        Optimizer parameters
        ----------------------------------
        :param optimizer_name:      String to identify the optimizer
        :param lr:                  The learning rate
        :param momentum:            Momentum factor for SGD, RMSProp, and LARS
        :param weight_decay:        Weight Decay for SGD, RMSProp, and LARS
        :param alpha:               Alpha for RMSProp
        :param eps:                 Epsilon for RMSProp or LARS
        :param trust_coef:          Trust coefficient for LARS

        Others
        -------------------------------------
        :param stopping_patience:   Number of epochs that val_loss is allowed to not improve before stopping
        :param subset:              Optional: If only a subset of the data should be used
        """

        if not hasattr(self, "train_ds"):
            self.tokenize_data()

        # Prepare model and optimizer
        model = self.check_for_existing_weights(epochs=epochs)
        optimizer = m.get_optimizer(params=model.parameters(), optimizer_name=optimizer_name, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, alpha=alpha, eps=eps, trust_coef=trust_coef)

        # Prepare early stopping
        self.stopping_patience = stopping_patience
        self.stagnant_epochs = 0

        # Create summary writer and a csv-file to write the loss values (if not wandb sweeping)
        if not self.use_wandb:
            writer = SummaryWriter(log_dir=self.log_path)

            if not self.resume_from_existing_model:
                with open(self.csv_path, 'w') as file:
                    file.write('epoch,train_loss,val_loss,\n')

        # Prepare the two dataloaders (the data is formatted for usage with torch and sent to the device)
        train_data = self.train_ds.select(range(subset)) if subset is not None else self.train_ds
        eval_data = self.eval_ds.select(range(subset)) if subset is not None else self.eval_ds
        train_dl = DataLoader(train_data.with_format("torch", device=device), batch_size=batch_size)
        eval_dl = DataLoader(eval_data.with_format("torch", device=device), batch_size=batch_size)

        print("\nPerforming training based on the following parameters:")
        print(f"- Epochs:           {epochs}")
        print(f"- Batchsize:        {batch_size}")
        print(f"- Num sentences:    {self.num_sentences}")
        print(f"- Optimizer:        {optimizer}")
        print(f"- Loss:             {self.loss}")
        print(f"- Patience:         {stopping_patience}\n\n")

        for epoch in range(self.start_epoch, epochs):
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

            # Print an update
            self.print_update(train_loss, val_loss)

            # Logging
            # If training is not part of wandb sweeping, log the results for tensorboard and as csv
            if not self.use_wandb:
                self.logging(writer, train_loss, val_loss, epoch)

            else:
                wandb.log({"train_loss": train_loss,
                           "val_loss": val_loss})

            # Perform checkpointing and check for early stopping
            if not self.continue_training_and_checkpoint(val_loss, model):
                print(f"No improvement on val_loss detected for {self.stopping_patience} epochs.")
                print("Stopping training...")
                break

        # Close the writer
        if not self.use_wandb:
            writer.flush()
            writer.close()



    def check_for_existing_weights(self, epochs):
        """
        Function to check if a model with the same weights was already trained.
        If so, the user is asked whether the training should be resumed and the necessary values are loaded.

        :param epochs:      How many epochs should be trained in total
        """
        model = self.model.to(device)
        self.previous_loss = float('inf')
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.resume_from_existing_model = False

        if os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            min_row = df[df["val_loss"] == df["val_loss"].min()]

            # Determine start_epoch and ask user for confirmation
            self.start_epoch = min_row["epoch"].values[0] + 1
            previous_epochs = df.shape[0]
            if  previous_epochs >= epochs:
                print(f"A model with the same name was already trained for {previous_epochs} epochs. "
                      f"Please choose a different model_name or delete the corresponding files in ./models/csv_logs, "
                      f"tensorboard_logs and weights.")
                exit(1)

            print(f"Existing logs were found under {self.csv_path}. "
                  f"Training would be resumed from epoch {self.start_epoch + 1}/{epochs}")
            resume_training = input("Would you like to continue training? (y/n): ").lower()
            if resume_training != "y":
                print("Training process aborted by user command.")
                exit(1)

            # Set loss values
            self.resume_from_existing_model = True
            self.previous_loss = self.best_val_loss = min_row["val_loss"].values[0]

            # Load the weights
            encoder_weights = T.load(self.weight_path)
            model.load_state_dict(encoder_weights)

            print(f"Existing weights loaded. Resuming training from epoch {self.start_epoch}.")

        return model


    def continue_training_and_checkpoint(self, val_loss, model):
        # Initialize the return value
        continue_training = True

        # Check if an improvement to the last epoch took place; If yes, reset stagnant epochs
        if val_loss < self.previous_loss:
            self.stagnant_epochs = 0

            # Check for new optimum; If yes, update the best_val_loss and checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                # Only checkpoint if not used in hyperparameter sweep
                if not self.use_wandb:
                    T.save(model.state_dict(), self.weight_path)
                    print(f"New checkpoint for validation loss. Model weights saved to {self.weight_path}\n")

        # Otherwise increase stagnant epochs and check patience
        else:
            self.stagnant_epochs += 1

            # If no improvement took place for the specified number of epochs, stop training
            if self.stagnant_epochs > self.stopping_patience:
                continue_training = False

        # Update the previous loss
        self.previous_loss = val_loss

        return continue_training



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



    def print_update(self, train_loss, val_loss):
        """
        Function to print an update based on training
        :param train_loss:          Loss on the training data
        :param val_loss:            Loss on the validation data
        """

        # Get the metrics into a string
        train_str = [str("train_loss = %10.8f | " % train_loss)]
        val_str = [str("val_loss   = %10.8f | " % val_loss)]

        print("".join(train_str))
        print("".join(val_str))




    def logging(self, writer, train_loss, val_loss, epoch):
        """
        Function to perform logging for Tensorboard and into a CSV-File
        :param writer:          Instance of torch.utils.tensorboard.SummaryWriter
        :param train_loss:      Loss on the training data
        :param val_loss:        Loss on the validation data
        :param epoch:           Current epoch
        :return:
        """

        # Create the outline for the CSV-File
        out_line = [str(epoch), str(train_loss), str(val_loss)]

        # Write the losses for tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        with open(self.csv_path, 'a') as file:
            file.write(",".join(out_line) + '\n')
