# 1. Take in a loss name (Pairwise, Triplet, InfoNCE) and load all three csv for all optimizers
# 2. Get the loss values for the validation data and train data for each optimizer
# 3. Create two plots (With the same y-axis): One for the validation data and one for the training data
#   - Use the same three colors the optimizers in each of the loss graphs
#   - (If necessary, use dotted line for validation and solid line for train)

# In the paper, put two graphs side by side for each loss
# Reuse code from distance visualization

import os
import pandas as pd
import math
import matplotlib.pyplot as plt

# =======================================================
# Constants
# =======================================================
CONTRASTIVE_LOSS_DIR = "../../contrastive/models/csv_logs/"
LOSS_NAMES = ["Pairwise", "Triplet", "InfoNCE"]
OPTIMIZER_RENAMING = {"SGD": "SGD",
                      "RMS": "RMSProp",
                      "LARS": "LARS"}
OPTIMIZER_COLORS = {"SGD": "darkturquoise",
                    "RMSProp": "darkmagenta",
                    "LARS": "darkorange"}
LINESTYLES = {"Train": "-",
              "Validation": "--"}



def save_optimizer_visualization(loss_name="Pairwise", linewidth=3, large_fontsize=18, small_fontsize=15):
    # Get the dictionary of loss series (if it is empty, end the function)
    loss_dict = get_optimizer_losses(loss_name=loss_name)
    if len(loss_dict) == 0:
        return

    # Define min and max values to set the y limits and num_epochs for the x axis
    max_loss = 0.0
    min_loss = float("inf")
    num_epochs = 0

    # Update num_epochs, min and max value
    for set_name, optimizer_dict in loss_dict.items():
        for optimizer_name, loss_series in optimizer_dict.items():
            # Update num_epochs, min and max value
            num_epochs = max(num_epochs, len(loss_series))
            max_loss = max(max_loss, max(loss_series))
            min_loss = min(min_loss, min(loss_series))

    # Determine the y limits by first finding the position of the first non-zero digit and then rounding to that
    max_round_digits = int(('%e' % max_loss).partition('-')[2])
    min_round_digits = int(('%e' % min_loss).partition('-')[2]) - 1
    y_max = round_decimals_up(max_loss, max_round_digits)
    y_min = round_decimals_down(min_loss, min_round_digits)

    # Plot the two graphs separately to ensure same plt configurations (did not work correctly otherwise)
    for set_name, optimizer_dict in loss_dict.items():
        # Create a new subplot
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        for optimizer_name, loss_series in optimizer_dict.items():
            color = OPTIMIZER_COLORS[optimizer_name]
            linestyle = LINESTYLES[set_name]
            loss_series.plot.line(ax=ax, linewidth=linewidth, linestyle=linestyle, color=color)

        # Set the configurations and save the result
        plt.ylabel("Loss", fontsize=large_fontsize)
        plt.ylim(y_min, y_max)
        plt.yticks(fontsize=small_fontsize)
        plt.xlabel("Epoch", fontsize=large_fontsize)
        plt.xticks(fontsize=large_fontsize)
        plt.legend(loc="upper right", fontsize=small_fontsize)

        fig.savefig(loss_name + "_" + set_name + ".png", dpi=300.0)




def get_optimizer_losses(loss_name="Pairwise"):
    # Get all filenames and restrict the list to those for the specified loss
    filenames = os.listdir(CONTRASTIVE_LOSS_DIR)
    relevant_filenames = [name for name in filenames if name.startswith(loss_name) and name.endswith(".csv")]

    # Create a dictionary to save the train and validation loss for each optimizer
    # Outer dictionary contains Key="Train" or "Validation". Inner dictionaries have series for each optimizer
    loss_dict = {}
    for file_name in relevant_filenames:
        optimizer_short = file_name.split("_")[1].split(".")[0]
        optimizer_name = OPTIMIZER_RENAMING[optimizer_short]

        # Get the train and validation loss from the df and rename them after the corrected optimizer_name
        df = pd.read_csv(CONTRASTIVE_LOSS_DIR+file_name)
        train_series = df["train_loss"].rename(optimizer_name)
        val_series = df["val_loss"].rename(optimizer_name)

        # Save the two series in their corresponding dictionary
        train_dict = loss_dict["Train"] if "Train" in loss_dict else {}
        val_dict = loss_dict["Validation"] if "Validation" in loss_dict else {}

        train_dict[optimizer_name] = train_series
        val_dict[optimizer_name] = val_series

        loss_dict["Train"] = train_dict
        loss_dict["Validation"] = val_dict

    return loss_dict



# =======================================================
# Rounding functions: https://kodify.net/python/math/round-decimals/
# =======================================================

def round_decimals_up(number: float, decimals=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor





# Define main usage
if __name__ == "__main__":
    # Loop over all loss names
    for name in LOSS_NAMES:
        save_optimizer_visualization(loss_name=name)
