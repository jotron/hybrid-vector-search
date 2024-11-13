import matplotlib.pyplot as plt
import csv
from collections import Counter

def read_csv_data(filename):
    """Reads a CSV file with a single row of numeric values."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data = [float(value) for value in row]  # Convert each value to float
    return data

def create_histogram(data, subplot, title):
    """Generates and saves a histogram from the data."""
    data = list(filter(lambda x: x != -1, data))
    plt.subplot(2, 2, subplot)
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

def create_value_plot(data, subplot, title):
    """Generates and saves a histogram from the data."""
    data = list(filter(lambda x: x != -1, data))
    plt.subplot(2, 2, subplot)
    frequency = Counter(data)
    sorted_items = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    values, counts = zip(*sorted_items)  # Separate the values and their counts
    # Create the bar plot with log scale on y-axis
    plt.bar(values, counts, color='skyblue', edgecolor='black')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    print(counts[:10])


# Main Script
png_filename = 'data_distribution.png'
plt.figure(figsize=(6, 5))

# Read data from CSV
node_timestamps = read_csv_data('node_timestamps.csv')
query_timestamps = read_csv_data('query_timestamps_l.csv') + read_csv_data('query_timestamps_r.csv')
node_values = read_csv_data('node_values.csv')
query_values = read_csv_data('query_values.csv')

# Create and save histogram
create_histogram(node_timestamps, 1, "Node timestamps")
create_histogram(query_timestamps, 2, "Query timestamps")
create_value_plot(node_values, 3, "Node values")
create_value_plot(query_values, 4, "Query values")

# Save histogram as PNG
plt.tight_layout()
plt.savefig(png_filename)
print(f"Histogram saved as {png_filename}")
plt.close()