import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
# Metrics data
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'MCC', 'NPV', 'FPR', 'FNR']
with_deit_values = [99.18256, 98.97628, 99.29769, 99.83956, 99.13331, 98.9983, 99.83028, 0.160444, 0.702313]
without_deit_values = [96.86649, 96.90659, 96.56756, 99.35175, 96.72957, 96.1555, 99.3643, 0.64825, 3.432443]

# Set up the bar width and position for bars
bar_width = 0.35
index = np.arange(len(metrics))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for 'With DeiT & dropout regularization' and 'Without DeiT & dropout regularization'
bar1 = ax.bar(index - bar_width/2, with_deit_values, bar_width, label='With DeiT & Dropout', color='steelblue')
bar2 = ax.bar(index + bar_width/2, without_deit_values, bar_width, label='Without DeiT & Dropout', color='lightblue')

# Add labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
# ax.set_title('Comparison of Metrics with and without DeiT & Dropout Regularization')
ax.set_xticks(index)
ax.set_xticklabels(metrics, rotation=45, ha='center')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
