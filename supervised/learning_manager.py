from datasets import load_dataset, ClassLabel, Value

from transformers import AutoTokenizer
import torch as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, AUROC, F1Score
from glob import glob
import wandb

import os
import time

import models as m

# ================================================================
# Constants
# ================================================================

DATASET = {"path": "glue", "name": "mrpc"}
MODEL_OUT_PATH = os.path.abspath('./models/')

# Path where the weights of the contrastive models are stored
PRETRAINED_PATH = os.path.abspath("../contrastive/models/weights/")

# ================================================================
# Preparation
# ================================================================
device = T.device("cuda" if T.cuda.is_available() else "cpu")


# ================================================================
# Main class
# ================================================================
class LearningManager():
    def __init__(self,  model_name=None, encoder="baseline", use_wandb=False):
        """
        Defines the instance of the LearningManager
        :param model_name:      Optional: Name to identify weights and logs; If None, a name is constructed
        :param encoder:         "baseline" or name of a pretrained contrastive model
                                (name refers to the weights stored in ../contrastive/model/weights)
        :param use_wandb:       True: Training is conducted as part of wandb sweeping
        """

        print("\n" + "="*50)
        print("       Supervised Learning ")
        print("="*50)

        print("Preparing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(m.MODEL_NAME)

        # Load the model as baseline from huggingface or weights from a contrastive pre-trained model.
        print("Preparing the model...")
        self.model = m.SupervisedModel()
        self.encoder = encoder
        print(f"Encoder: {encoder}")
        if encoder != "baseline":
            self.load_encoder_weights(encoder)

        self.loss = T.nn.BCEWithLogitsLoss()

        if model_name is None:
            time_stamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            model_name = m.MODEL_NAME.split("/")[1] + "_" + time_stamp

        # Set the metrics:
        self.metrics = {"accuracy": Accuracy(num_classes=1).to(device),
                        "auroc": AUROC(num_classes=1).to(device),
                        "f1": F1Score(num_classes=1).to(device)}

        # Defining the path
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


    def load_encoder_weights(self, encoder):
        try:
            # Get the encoder weights
            encoder_weights = T.load(PRETRAINED_PATH + "/" + encoder + ".pt")

        except:
            print(f"No weights were found for the encoder '{encoder}' in {PRETRAINED_PATH}")
            exit(1)

        # Adapt the encoder weights to fit the supervised model
        model_state_dict = self.model.state_dict()
        encoder_weights['linear.weight'] = model_state_dict['linear.weight']
        encoder_weights['linear.bias'] = model_state_dict['linear.bias']

        self.model.load_state_dict(encoder_weights)

        # Freeze the encoder to only train the classifier
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def load_dataset(self):
        """
        Function to load an already preprocessed dataset from csv.
        Format shold be:
        sentence1, sentence2, label
        """
        self.dataset = load_dataset("JoPro/supervised_paraphrases", use_auth_token=True)

        self.dataset = self.dataset.cast_column("label",ClassLabel(num_classes=2))


    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    def load_dataset_glue(self):
        """
        Function that sets the dataset-attribute
        :param dataset_name:    Key in the DATASETS-dictionary
        """
        self.dataset = load_dataset(**DATASET)



    # Todo: Adapt this to our dataset
    # Source: https://huggingface.co/docs/transformers/training
    def tokenize_data(self):
        """
        Function to tokenize the attribute dataset and split it into train_ds, eval_ds and test_ds
        """

        if not hasattr(self, "dataset"):
            self.load_dataset()

        # Determine removal columns (index and all sentences)
        remove_columns = ["index", "sentence1", "sentence2", 'path', 'name', 'split']

        # Apply tokenization and remove unnecessary columns
        tokenized_ds = self.dataset.map(lambda example: self.tokenize_function(example),
                                        remove_columns=remove_columns)

        tokenized_ds = tokenized_ds.rename_column("label", "labels")

        tokenized_ds = tokenized_ds.with_format("torch")
        print(tokenized_ds.column_names)
        # Unsqueeze the label column
        tokenized_ds = tokenized_ds.map(lambda example: {"labels": T.unsqueeze(example["labels"], dim=0)},
                                        remove_columns=["labels"])

        # Assign the splits
        self.train_ds = tokenized_ds["train"]
        self.eval_ds = tokenized_ds["validation"]
        self.test_ds = tokenized_ds["test"]


    def tokenize_function(self, example):
        # Get the two sentences
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]

        # Return the tokenized sentences (Note: They are in one array)
        # Pad to the maximum length of the model
        return self.tokenizer(sentence1, sentence2, padding="max_length", truncation=True)




    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def conduct_training(self, epochs=15, batch_size=32, optimizer_name='sgd', lr=0.01, momentum=0, weight_decay=0,
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
        :param stopping_patience:  Number of epochs that val_loss is allowed to not improve before stopping
        :param subset:              Optional: If only a subset of the data should be used
        """

        if not hasattr(self, "train_ds"):
            self.tokenize_data()

        # Prepare model and optimizer
        model = self.model.to(device)
        optimizer = m.get_optimizer(params=model.parameters(), optimizer_name=optimizer_name, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, alpha=alpha, eps=eps, trust_coef=trust_coef)

        # Prepare early stopping and checkpointing
        self.stopping_patience = stopping_patience
        self.stagnant_epochs = 0
        self.previous_loss = float('inf')
        self.best_val_loss = float('inf')

        # Create summary writer and a csv-file to write the loss values (if not wandb sweeping)
        if not self.use_wandb:
            writer = SummaryWriter(log_dir=self.log_path)
            with open(self.csv_path, 'w') as file:
                # Fill a list with strings for the header
                out_line = ["epoch", "train_loss"]
                for name in self.metrics.keys():
                    out_line.append("train_" + name)
                out_line.append("val_loss")
                for name in self.metrics.keys():
                    out_line.append("val_" + name)

                file.write(",".join(out_line) + "\n")

        # Prepare the two dataloaders (the data is formatted for usage with torch and sent to the device)
        train_data = self.train_ds.select(range(subset)) if subset is not None else self.train_ds
        eval_data = self.eval_ds.select(range(subset)) if subset is not None else self.eval_ds
        train_dl = DataLoader(train_data.with_format("torch", device=device), batch_size=batch_size)
        eval_dl = DataLoader(eval_data.with_format("torch", device=device), batch_size=batch_size)


        print("\nPerforming training based on the following parameters:")
        print(f"- Encoder:          {self.encoder}")
        print(f"- Epochs:           {epochs}")
        print(f"- Batchsize:        {batch_size}")
        print(f"- Optimizer:        {optimizer}")
        print(f"- Loss:             {self.loss}")
        print(f"- Patience:         {stopping_patience}\n\n")

        for epoch in range(epochs):
            print("\n" + "-" * 100)
            print(f"Epoch {epoch+1}/{epochs}")

            with T.enable_grad():
                # Set the model into train mode
                model.train()
                train_loss, train_metrics = self.loss_epoch(model=model, dataloader=train_dl, optimizer=optimizer)

            # Perform evaluation
            model.eval()
            with T.no_grad():
                val_loss, val_metrics = self.loss_epoch(model=model, dataloader=eval_dl)

            # Change the keys in the metric-dicts to reflect whether they are from the train or val set
            for key in self.metrics.keys():
                train_metrics["train_" + key] = train_metrics.pop(key)
                val_metrics["val_" + key] = val_metrics.pop(key)

            # Print an update
            self.print_update(train_loss, val_loss, train_metrics, val_metrics)

            # Logging
            # If training is not part of wandb sweeping, log the results for tensorboard and as csv
            if not self.use_wandb:
                self.logging(writer, train_loss, val_loss, train_metrics, val_metrics, epoch)

            else:
                wandb_dict = {**{"train_loss": train_loss, "val_loss": val_loss}, **train_metrics, **val_metrics}
                wandb.log(wandb_dict)

            # Perform checkpointing and check for early stopping
            if not self.continue_training_and_checkpoint(val_loss, model):
                print(f"No improvement on val_loss detected for {self.stopping_patience} epochs.")
                print("Stopping training...")
                break

        # Close the writer
        if not self.use_wandb:
            writer.flush()
            writer.close()


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

        # Initialize empty tensors to store labels and logits for metric calculation
        epoch_labels = T.empty(size=(0, 1), device=device, dtype=T.int32)
        epoch_logits = T.empty(size=(0, 1), device=device, dtype=T.float32)

        # Loop over all batches in the data
        for batch in dataloader:
            # Get the labels
            labels = batch['labels']

            # Get the logits from the batch
            logits = model(batch=batch)

            # Update epoch tensors
            epoch_labels = T.cat((epoch_labels, labels), 0)
            epoch_logits = T.cat((epoch_logits, logits), 0)

            # Compute the loss value based on the labels and logits; Optimizer is passed in case of usage with train
            loss_val = self.loss_batch(logits=logits, labels=labels, optimizer=optimizer)

            # Update the running loss
            ep_loss += loss_val

        # Get the epoch values for all the metrics
        epoch_metrics = self.metrics_epoch(logits=epoch_logits, labels=epoch_labels)

        # Return the normalized loss and the metrics
        return (ep_loss / ds_len), epoch_metrics


    def loss_batch(self, logits, labels, optimizer=None):
        """
        Function to calculate the loss on one batch
        :param logits:          The logits of the current batch
        :param labels:          The labels for each of the sentence pairs
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the batch
        """

        labels = labels.float()
        loss = self.loss(logits, labels)

        if optimizer is not None:
            # Reset the gradients, compute new ones and perform a weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def metrics_epoch(self, logits, labels):
        """
        Function to calculate the metrics for the current epoch
        :param logits:      The logits of the current epoch
        :param labels:      The labels for each of the sentence pairs
        :return:            The metric values in a dictionary
        """

        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(logits, labels)

        return results



    def print_update(self, train_loss, val_loss, train_metrics, val_metrics):
        """
        Function to print an update based on training
        :param train_loss:          Loss on the training data
        :param val_loss:            Loss on the validation data
        :param train_metrics:       Dictionary of metrics achieved on the training data
        :param val_metrics:         Dictionary of metrics achieved on the validation data
        """

        # Get the metrics into a string
        train_str = [str("train_loss = %10.8f | " % train_loss)]
        for name, metric in train_metrics.items():
            train_str.append(name + " = ")
            train_str.append("%10.8f | " % metric)

        val_str = [str("val_loss   = %10.8f | " % val_loss)]
        for name, metric in val_metrics.items():
            val_str.append(name + "   = ")
            val_str.append("%10.8f | " % metric)

        print("".join(train_str))
        print("".join(val_str))




    def logging(self, writer, train_loss, val_loss, train_metrics, val_metrics, epoch):
        """
        Function to perform logging for Tensorboard and into a CSV-File
        :param writer:          Instance of torch.utils.tensorboard.SummaryWriter
        :param train_loss:      Loss on the training data
        :param val_loss:        Loss on the validation data
        :param train_metrics:   Dictionary of metrics achieved on the training data
        :param val_metrics:     Dictionary of metrics achieved on the validation data
        :param epoch:           Current epoch
        :return:
        """

        # Create the outline for the CSV-File
        out_line = [str(epoch), str(train_loss)]

        # Write the losses for tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Loop over the metrics, write them for Tensorboard and append them to the out_line
        for name, metric in train_metrics.items():
            metric_item = metric.item()
            writer.add_scalar(str(name.split("_")[1] + "/train"), metric_item, epoch)
            out_line.append(str(metric_item))

        out_line.append(str(val_loss))
        for name, metric in val_metrics.items():
            metric_item = metric.item()
            writer.add_scalar(str(name.split("_")[1] + "/val"), metric_item, epoch)
            out_line.append(str(metric_item))


        with open(self.csv_path, 'a') as file:
            file.write(",".join(out_line) + '\n')