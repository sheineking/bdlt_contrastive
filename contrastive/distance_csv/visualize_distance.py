import pandas as pd
import matplotlib.pyplot as plt
import argparse

CSV_NAME = "pairwise_eps3.csv"
STEP_SIZE = 2

def vis_distance(csv_name, step_size):
    df = pd.read_csv(csv_name)

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

