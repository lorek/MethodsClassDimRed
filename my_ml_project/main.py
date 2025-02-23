import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import pickle

# Import from our modules (assuming these files are inside the /models folder)
from models.my_functions import sum_of_squares
from models.linear_regression import myLinearRegression1D
from models.my_utils import generate_data


## Typical usage:
## With default options:
##    python main.py
## Generate data:
##    python main.py --generate_data True
## With some extra options:
##    python main.py --n_squares 12 --generate_data True --show_plots True


def ParseArguments():
    parser = argparse.ArgumentParser(description="My ML Project")
    parser.add_argument('--n_squares', default="10",
                        help='For computing 1^2+2^2+...+n_squares^2 (default: %(default)s)')
    parser.add_argument('--data_csv_file', default="data/sample_data.csv",
                        help='Data CSV file for regression (default: %(default)s)')
    parser.add_argument('--data_pkl_file', default="data/sample_data.pkl",
                        help='Data pickle file for regression (default: %(default)s)')
    parser.add_argument('--data_path', default="data/",
                        help='Data folder (default: %(default)s)')
    parser.add_argument('--generate_data', default="False",
                        help='Generate data? (default: %(default)s)')
    parser.add_argument('--show_plots', default="True",
                        help='Show plots? (default: %(default)s)')
    args = parser.parse_args()
    return args



def main():
    # Parse command-line arguments
    args = ParseArguments()
    n_squares = int(args.n_squares)
    data_csv_file = args.data_csv_file
    data_pkl_file = args.data_pkl_file
    data_path = args.data_path
    generate_data_bool = args.generate_data.lower() == "true"
    show_plots = args.show_plots.lower() == "true"

    print("Sum of squares for n =", n_squares, ":", sum_of_squares(n_squares))

    if generate_data_bool:
        print("Generating data...")
        generate_data(data_path)


    ## we will use only pickle data, but this is how to read .csv
    # df = pd.read_csv(data_csv_file)
    # a_true = 2.0
    # b_true = 1.0
    # x_data = df["x_data"].values.reshape(-1, 1)
    # y_data = df["y_data"].values.reshape(-1, 1)

    # Read data from pickle
    if os.path.exists(data_pkl_file):
        with open(data_pkl_file, "rb") as f:
            print("Reading", data_pkl_file)
            data = pickle.load(f)
    else:
        print(f"\nFile {data_pkl_file} does not exist. Generate data with:")
        print("python main.py --generate_data True")
        print("Exiting...")
        quit()

    x_data = data["x_data"]
    y_data = data["y_data"]
    a_true = data["a_true"]
    b_true = data["b_true"]

    # Create and train the model using the full data
    model = myLinearRegression1D()
    print("Fitting data...")
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)

    print("True slope (a):\t\t", a_true, "  Trained slope (beta1):\t\t", model.beta1)
    print("True intercept (b):\t", b_true, "  Trained intercept (beta0):\t", model.beta0)

    if show_plots:
        print("Creating plots...")
        plt.figure(figsize=(10,7))
        plt.scatter(x_data, y_data, alpha=0.3, label='Training data')
        plt.plot(x_data, y_pred, color='red', label='Fitted regression line')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("1D Linear Regression Fit")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
