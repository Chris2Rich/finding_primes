import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

results_dir = "times"
files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

plt.figure(figsize=(12, 7))

for file_name in files:
    file_path = os.path.join(results_dir, file_name)
    
    data_rows_df = pd.read_csv(file_path, skiprows=1, header=None)

    x_labels_str_series = data_rows_df.iloc[:, 0].astype(str)
    
    parsed_exponents = []
    for label in x_labels_str_series:
        try:
            start_index = label.index("10^") + 3 
            end_index = label.find(",", start_index)
            if end_index == -1: 
                exponent_str = label[start_index:]
            else:
                exponent_str = label[start_index:end_index]
            parsed_exponents.append(int(exponent_str))
        except ValueError: 
            parsed_exponents.append(np.nan) 

    x_values_series = pd.Series(parsed_exponents, dtype=float)
    
    time_data_columns = data_rows_df.iloc[:, 1:]
    numeric_time_data = time_data_columns.apply(pd.to_numeric, errors="coerce")
    y_values_series = numeric_time_data.mean(axis=1)
    
    valid_indices_mask = ~x_values_series.isna() & ~y_values_series.isna()
    final_x_values = x_values_series[valid_indices_mask]
    final_y_values = y_values_series[valid_indices_mask]

    if not final_x_values.empty:
        plt.plot(final_x_values, final_y_values, label=file_name.replace(".csv",""), marker="o", markersize=4)

plt.yscale("log")
plt.ylim(bottom=1)
plt.xlabel("First 10^N primes")
plt.ylabel("Average Time (microseconds, log scale)")
plt.title("Average Time vs. N (Logarithmic Scale)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("average_time_vs_n.png", dpi=300)