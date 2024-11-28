import os

import matplotlib.pyplot as plt

# Directory containing the data files
data_dir = r'C:\Users\kid\labQT\Lab2024\SINGLE_PHOTON\QTLab2425\IRdetection\Tests\power_combiner_runs'

# Initialize a figure
fig = plt.figure()
n_columns = 5
column_index = 1
row_index = 1

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('near.txt'):  # Assuming the data files are .txt files
        filepath = os.path.join(data_dir, filename)
        data_x = []
        data_y = []
        with open(filepath, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                data_x.append(x)
                data_y.append(y)
        # Plot the data
        
        #fig.add_subplot(row_index)
        plt.plot(data_x, data_y)
        plt.axhline(15, linestyle='--', color="red")
        plt.axhline(7.5, linestyle='--', color="blue")
        # Add labels and legend
        plt.xlabel('frequency')
        plt.ylabel('Power')
        plt.title('Power Combiner Runs')
        row_index += 1
        column_index += 1        
        column_index = column_index % n_columns




plt.legend()

# Show the plot
plt.show()