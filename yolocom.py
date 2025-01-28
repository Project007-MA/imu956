import numpy as np
import matplotlib.pyplot as plt

# Configure plot style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data
classifiers = [
    "YOLO 3",
    "YOLO 4",
    "YOLO 5",
    "YOLO 7",
    "Proposed"
]

metrics = ["Precision", "Recall", "F1 Score", "mAP"]
values = [
    [0.654, 0.694, 0.663, 0.721],  # YOLO v3 + DarkNet-53
    [0.681, 0.722, 0.681, 0.754],  # YOLO v4 + CSPDarkNet53
    [0.702, 0.702, 0.693, 0.734],  # YOLO v5 + CSPDarkNet53
    [0.741, 0.771, 0.792, 0.791],  # YOLO v7 + ELAN-Net
    [0.989, 0.993, 0.991, 0.984],  # DeiT-YOLO v8 (Proposed)
]

# Colors for each metric
colors =['steelblue', 'skyblue', 'green', 'mediumpurple']

# Plot settings
bar_width = 0.15
x = np.arange(len(classifiers))  # X-axis positions

# Create the bar plot
plt.figure(figsize=(12, 7))

for i, (metric, color) in enumerate(zip(metrics, colors)):
    plt.bar(
        x + i * bar_width, 
        [values[j][i] for j in range(len(classifiers))], 
        width=bar_width, 
        label=metric, 
        color=color
    )

# Add labels, title, and legend
plt.xlabel("Methods", labelpad=15)
plt.ylabel("Values", labelpad=15)
plt.xticks(x + (len(metrics) - 1) * bar_width / 2, classifiers)  # Center xticks under grouped bars
plt.legend(loc='upper left')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
