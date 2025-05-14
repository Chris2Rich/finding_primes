import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

results_dir = 'results'
files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

plt.figure(figsize=(12, 7)) 

for file in files:
    path = os.path.join(results_dir, file)
    try:
        df = pd.read_csv(path, header=0) 
        df_data = pd.read_csv(path, skiprows=1, header=None) 

        numeric_cols = df_data.iloc[:, 1:]
        numeric_cols = numeric_cols.apply(pd.to_numeric, errors='coerce')
        row_means = numeric_cols.mean(axis=1)
        x_values = range(len(row_means))

        plt.plot(x_values, row_means, label=file.replace('.csv',''), marker='o', markersize=4)

    except Exception as e:
        print(f"Could not process file {file}: {e}")

plt.yscale('log')
plt.ylim(bottom=1) 

plt.xlabel('Index of N (e.g., for 10^0, 10^1, ...)') 
plt.ylabel('Average Time (microseconds, log scale)')
plt.title('Average Time vs. N (Logarithmic Scale)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.show()