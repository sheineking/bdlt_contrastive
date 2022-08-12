import pandas as pd
import matplotlib.pyplot as plt
import argparse

CSV_NAME = "pairwise_eps3.csv"
CSV_PATH = "../../contrastive/distance_csv/"
STEP_SIZE = 2
NUM_TRAIN = 3668
NUM_VAL = 408
NUM_RECORDS = NUM_TRAIN + NUM_VAL


def label_epoch_and_set(df):
    # Assign the epoch numbers and validation or train set
    num_epochs = df.shape[0] // NUM_RECORDS
    for i in range(num_epochs):
        start_index = i*NUM_RECORDS
        val_start = start_index + NUM_TRAIN

        df.loc[df.index >= start_index, "epoch"] = i
        df.loc[df.index >= start_index, "set"] = "train"
        df.loc[df.index >= val_start, "set"] = "val"

    return df


def vis_epoch_distance(csv_name):
    df = pd.read_csv(CSV_PATH + csv_name)

    # Group by set, epoch and label
    df2 = df.groupby(["set", "epoch", "label"], as_index=False)["dist"].mean()

    # Create four series (one for each combination of label and set)
    train_0 = df2[(df2["label"] == 0) & (df2["set"] == "train")].set_index("epoch")["dist"]
    train_1 = df2[(df2["label"] == 1) & (df2["set"] == "train")].set_index("epoch")["dist"]
    val_0 = df2[(df2["label"] == 0) & (df2["set"] == "val")].set_index("epoch")["dist"]
    val_1 = df2[(df2["label"] == 1) & (df2["set"] == "val")].set_index("epoch")["dist"]

    # Rename the series
    train_0.rename("Train (Label=0)", inplace=True)
    train_1.rename("Train (Label=1)", inplace=True)
    val_0.rename("Val (Label=0)", inplace=True)
    val_1.rename("Val (Label=1)", inplace=True)

    # Define the title
    #eps = float(csv_name.split("eps")[1].split(".")[0]) / 10
    #title = f"Margin = {eps}"

    # Create the plot
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    #ax.set_title(title, fontsize=15)

    # Visualize the development
    train_0.plot.line(ax=ax, linewidth=4, linestyle="-", color="black")
    train_1.plot.line(ax=ax, linewidth=4, linestyle="--", color="black")
    val_0.plot.line(ax=ax, linewidth=4, linestyle="-", color="orange")
    val_1.plot.line(ax=ax, linewidth=4, linestyle="--", color="orange")

    plt.ylabel("Distance", fontsize=23)
    plt.ylim(0, 0.6)
    plt.yticks(fontsize=19)
    plt.xlabel("Epoch", fontsize=23)
    plt.xticks(fontsize=19)
    plt.legend(loc="upper left", fontsize=19)


    plt.show()


def vis_distance(csv_name, step_size):
    df = pd.read_csv(CSV_PATH + csv_name)

    # Assign timestamps
    length = len(df)
    timestamps = [i//step_size for i in range(length)]
    df["time"] = timestamps

    # Get the average for each label per timestamp
    series0 = df[df["label"] == 0].groupby("time")["dist"].mean()
    series1 = df[df["label"] == 1].groupby("time")["dist"].mean()
    series0.rename("label0", inplace=True)
    series1.rename("label1", inplace=True)

    # Put them into one df and apply forward fill to avoid missing values
    df_graph = pd.merge(series0, series1, how="outer", left_index=True, right_index=True)
    df_graph = df_graph.ffill()

    # Define the title
    eps = float(csv_name.split("eps")[1].split(".")[0])/10
    title = f"Distance development for eps={eps}; One timestep equals {step_size} examples"

    # Visualize the development
    df_graph.plot.line()
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distance visualization')

    # First, define the mode (contrastive or supervised) and base configuration to be used
    parser.add_argument('--csv_name', metavar='csv_name', type=str, required=False,
                        help='The CSV-File containing the distance values')
    parser.add_argument('--step_size', metavar='step_size', type=int, required=False,
                        help='Number of elements over which to average the distance values')

    args = parser.parse_args()

    CSV_NAME = args.csv_name if args.csv_name is not None else CSV_NAME
    STEP_SIZE = args.step_size if args.step_size is not None else STEP_SIZE

    vis_distance(CSV_NAME, STEP_SIZE)

