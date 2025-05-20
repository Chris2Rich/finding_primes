import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

results_dir = "times"
# List all subdirectories in results_dir
folders = [d for d in os.listdir(results_dir)
           if os.path.isdir(os.path.join(results_dir, d))]

plt.figure(figsize=(12, 7))

for folder in folders:
    folder_path = os.path.join(results_dir, folder)
    # Find all CSV files in this folder
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        continue

    # Read all CSVs into a list of DataFrames
    dfs = [pd.read_csv(f, skiprows=1, header=None) for f in csv_files]

    # Parse x-values (exponents) from the first CSV's first column
    labels = dfs[0].iloc[:, 0].astype(str)
    exponents = []
    for label in labels:
        try:
            start = label.index("10^") + 3
            end = label.find(",", start)
            exp_str = label[start:] if end == -1 else label[start:end]
            exponents.append(int(exp_str))
        except ValueError:
            exponents.append(np.nan)
    x_values = pd.Series(exponents, dtype=float)

    # For each DataFrame, compute the mean time per row across its time columns
    series_list = []
    for df in dfs:
        times = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        series_list.append(times.mean(axis=1))

    # Combine into a DataFrame and average across CSVs
    time_df = pd.DataFrame(series_list).T
    y_values = time_df.mean(axis=1)

    # Filter out invalid entries
    mask = ~x_values.isna() & ~y_values.isna()
    x_final = x_values[mask]
    y_final = y_values[mask]

    # Plot one line per folder
    plt.plot(x_final, y_final, label=folder, marker='o', markersize=4)

# Final formatting
plt.yscale("log")
plt.ylim(bottom=1)
plt.xlabel("First 10^N primes")
plt.ylabel("Average Time (microseconds, log scale)")
plt.title("Average Time vs. N for each method (Log Scale)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("average_time_vs_n_by_folder.png", dpi=300)