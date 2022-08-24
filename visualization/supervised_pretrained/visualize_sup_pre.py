import os
import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt

# =======================================================
# Constants
# =======================================================
MODELS = {"pretrained": {"dir": "../../pretrained/models/csv_logs/",
                         "models": ["Pretrained_Pairwise.csv", "Pretrained_Triplet.csv", "Pretrained_InfoNCE.csv"]},
          "supervised": {"dir": "../../supervised/models/csv_logs/",
                         "models": ["Supervised_SGD.csv", "Supervised_RMS.csv", "Supervised_LARS.csv"]}}

RENAMING_DICT = {"Pretrained_Pairwise.csv": "Pre_Pairwise",
                 "Pretrained_Triplet.csv": "Pre_Triplet",
                 "Pretrained_InfoNCE.csv": "Pre_InfoNCE",
                 "Supervised_SGD.csv": "Sup_SGD",
                 "Supervised_RMS.csv": "Sup_RMSProp",
                 "Supervised_LARS.csv": "Sup_LARS"}

VALUE_DICT = {"loss": {"Label": "Loss",
                       "Legend_Pos": "below"},
              "auroc": {"Label": "AUROC",
                       "Legend_Pos": "lower right"},
              "accuracy": {"Label": "Accuracy",
                           "Legend_Pos": "lower right"},
              "f1": {"Label": "F1-Score",
                     "Legend_Pos": "lower right"}
              }

OPTIMIZER_COLORS = {"Pre_Pairwise": "darkgreen",
                    "Pre_Triplet": "darkred",
                    "Pre_InfoNCE": "mediumblue",
                    "Sup_SGD": "darkturquoise",
                    "Sup_RMSProp": "darkmagenta",
                    "Sup_LARS": "darkorange"}

LINESTYLES = {"Train": "-",
              "Validation": "--"}



def save_visualization(value_name="loss", model_types=None, linewidth=2.5, large_fontsize=16, small_fontsize=14):
    if model_types is None:
        model_types = MODELS.keys()

    # Get the dictionary for each model (if it is empty, end the function)
    main_dict = {"Train": {}, "Validation": {}}
    for type in model_types:
        type_dict = get_model_values(value_name=value_name, model_type=type)
        main_dict["Train"] = {**main_dict["Train"], **type_dict["Train"]}
        main_dict["Validation"] = {**main_dict["Validation"], **type_dict["Validation"]}

    if len(main_dict) == 0:
        return


    # Define min and max values to set the y limits and num_epochs for the x axis
    max_val = 0.0
    min_val = float("inf")
    num_epochs = 0

    # Update num_epochs, min and max value
    for set_name, model_dict in main_dict.items():
        for line_name, series in model_dict.items():
            # Update num_epochs, min and max value
            num_epochs = max(num_epochs, len(series))
            max_val = max(max_val, max(series))
            min_val = min(min_val, min(series))

    # Determine the y limits by first finding the position of the first non-zero digit and then rounding to that
    max_round_digits = int(('%e' % max_val).partition('-')[2]) + 1
    min_round_digits = int(('%e' % min_val).partition('-')[2]) + 1
    y_max = round_decimals_up(max_val, max_round_digits)
    y_min = round_decimals_down(min_val, min_round_digits)

    # Plot the two graphs separately to ensure same plt configurations (did not work correctly otherwise)
    for set_name, model_dict in main_dict.items():
        # Create a new subplot
        fig, ax = plt.subplots(figsize=(9, 9))

        for line_name, series in model_dict.items():
            color = OPTIMIZER_COLORS[line_name]
            linestyle = LINESTYLES[set_name]
            series.plot.line(ax=ax, linewidth=linewidth, linestyle=linestyle, color=color)

        # Set the configurations and save the result
        y_label = VALUE_DICT[value_name]["Label"]
        legend_pos = VALUE_DICT[value_name]["Legend_Pos"]

        plt.ylabel(y_label, fontsize=large_fontsize)
        plt.ylim(y_min, y_max)
        plt.yticks(fontsize=small_fontsize)
        plt.xlabel("Epoch", fontsize=large_fontsize)
        plt.xticks(fontsize=large_fontsize)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   fancybox=True, shadow=True, ncol=3, fontsize=small_fontsize)

        fig.savefig(set_name + "_" + y_label + ".png", dpi=300.0)




def get_model_values(value_name, model_type="supervised"):
    # Get all filenames and restrict the list to those for the specified model type
    filenames = os.listdir(MODELS[model_type]["dir"])
    relevant_filenames = [name for name in filenames if name in MODELS[model_type]["models"]]

    # Create a dictionary to save the train and validation values for each model
    # Outer dictionary contains Key="Train" or "Validation". Inner dictionaries have series for model
    model_dict = {}
    for file_name in relevant_filenames:
        line_name = RENAMING_DICT[file_name]

        # Get the train and validation value from the df and rename them after the corrected line_name
        df = pd.read_csv(MODELS[model_type]["dir"]+file_name)
        train_series = df["train_" + value_name].rename(line_name)
        val_series = df["val_" + value_name].rename(line_name)

        # Save the two series in their corresponding dictionary
        train_dict = model_dict["Train"] if "Train" in model_dict else {}
        val_dict = model_dict["Validation"] if "Validation" in model_dict else {}

        train_dict[line_name] = train_series
        val_dict[line_name] = val_series

        model_dict["Train"] = train_dict
        model_dict["Validation"] = val_dict

    return model_dict



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
    parser = argparse.ArgumentParser(description='Visualization for Supervised and Pretrained')

    # First, define the mode (contrastive, supervised or pretrained) and base configuration to be used
    parser.add_argument('--value_name', metavar='value_name', type=str, required=True,
                        help='"loss", "auroc", "accuracy", or "f1". Chooses which kind of value to visualize.')

    args = parser.parse_args()

    # Create the two graphs
    save_visualization(value_name=args.value_name, model_types=["supervised", "pretrained"])
