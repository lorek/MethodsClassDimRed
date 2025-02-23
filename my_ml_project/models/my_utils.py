# models/my_utils.py

import numpy as np
import pandas as pd
import os
import pickle


def generate_data(data_path: str, a_slope: float = 2.0, b_intercept: float = 1.0, n_samples: int = 200):
    """
    Generate synthetic 1D data and save it as both CSV and pickle files.

    Parameters:
        data_path (str): Path to the folder where data files will be saved.
        a_slope (float): True slope.
        b_intercept (float): True intercept.
        n_samples (int): Number of data points.
    """
    np.random.seed(42)
    x_data = np.random.rand(n_samples, 1)
    y_data = b_intercept + a_slope * x_data + 0.1 * np.random.randn(n_samples, 1)

    # Save data as CSV
    xy_data = np.hstack([x_data, y_data])
    df = pd.DataFrame(xy_data, columns=["x_data", "y_data"])
    os.makedirs(data_path, exist_ok=True)
    csv_path = os.path.join(data_path, "sample_data.csv")
    print("Saving CSV to", csv_path)
    df.to_csv(csv_path, index=False)

    # Save data and parameters as a pickle file
    data_dict = {
        "x_data": x_data,
        "y_data": y_data,
        "a_true": a_slope,
        "b_true": b_intercept
    }
    pkl_path = os.path.join(data_path, "sample_data.pkl")
    print("Saving pickle to", pkl_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f)
