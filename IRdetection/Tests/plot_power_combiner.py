import os

import matplotlib.pyplot as plt

# Directory containing the data files
data_dir = r'C:\Users\kid\labQT\Lab2024\SINGLE_PHOTON\QTLab2425\IRdetection\Tests\power_combiner_runs'

# Loop through all files in the directory
all_data_x = []
all_data_y = []
file_names = []
for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):  # Assuming the data files are .txt files
        filepath = os.path.join(data_dir, filename)
        file_names.append(filename[15::])
        data_x = []
        data_y = []
        with open(filepath, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                data_x.append(x/1e9)  # Convert to GHz
                data_y.append(y)
                
        all_data_x.append(data_x)
        all_data_y.append(data_y)
        

# Plot the data
n_columns = 4
column_index = 1
row_index = 1
n_rows = -(-len(all_data_x) // n_columns)  # Ceiling division to get the upper value
fig, ax = plt.subplots(n_rows, n_columns, figsize=(15, 25))
for i in range(len(all_data_x)):
    ax[row_index-1, column_index-1].plot(all_data_x[i], all_data_y[i], label=file_names[i])
    # Add horizontal line at 7.5 dBm and 15 dBm
    ax[row_index-1, column_index-1].axhline(y=7.5, color='r', linestyle='--', label='7.5 dBm')
    ax[row_index-1, column_index-1].axhline(y=15, color='g', linestyle='--', label='15 dBm')
    ax[row_index-1, column_index-1].set_title(file_names[i])
    ax[row_index-1, column_index-1].set_xlabel('Frequency (GHz)')
    ax[row_index-1, column_index-1].set_ylabel('Power (dBm)')
    # Set x-axis limits
    ax[row_index-1, column_index-1].set_xlim(min(all_data_x[i]), max(all_data_x[i]))
    column_index += 1
    if column_index > n_columns:
        column_index = 1
        row_index += 1

plt.tight_layout(pad=2.0, h_pad=10)  # Adjust padding to reduce overlap, with increased vertical padding
plt.show()