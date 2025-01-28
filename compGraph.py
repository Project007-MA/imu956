import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data
models = ["CapsNet", "DensNet 121", "Leaky-SENet", "Proposed"]
accuracy = [95, 95, 92, 99.2]
precision = [97, 94.5, 90, 98.97]
recall = [95, 94.8, 88, 99.3]
f1_score = [96, 94.5, 89, 99.13]

# Bar width and positions
bar_width = 0.2
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting
plt.figure(figsize=(12, 7))
plt.bar(r1, accuracy, color='#F2B28C', width=bar_width, label='Accuracy')
plt.bar(r2, precision, color='#D2665A', width=bar_width, label='Precision')
plt.bar(r3, recall, color='#A94A4A', width=bar_width ,label='Recall')
plt.bar(r4, f1_score, color='#BE3144', width=bar_width,label='F1 Score')

# Labels and titles
plt.xlabel('Methods', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
# plt.title('Performance Comparison of Models', fontweight='bold')
plt.xticks([r + 1.5 * bar_width for r in range(len(models))], models)
plt.legend(framealpha=0.5)
plt.tight_layout()

# Show plot
plt.show()
