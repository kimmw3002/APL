import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python histogram.py <data_file>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    first_line = f.readline().strip()

data = np.array([float(x) for x in first_line.split('\t')])

plt.figure(figsize=(10, 6))
plt.hist(data, bins='auto', edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title(f'Histogram of first row — {sys.argv[1]}')
plt.tight_layout()
plt.show()
