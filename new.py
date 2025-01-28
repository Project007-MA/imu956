import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data
metrics = [
    "Accuracy", "Precision", "Recall", "Specificity", "F1 Score", 
    "MCC", "NPV", "FPR", "FNR"
]
values = [
    96.866485, 96.906595, 96.567557, 99.351750, 96.729567, 
    96.155497, 99.364301, 0.648250, 3.432443
]

# Plotting
plt.figure(figsize=(12, 8))
colors = ['#4c72b0' if val >= 90 else '#c44e52' for val in values]  # Professional colors: Blue for high, red for low
plt.barh(metrics, values, color=colors)

# Labels and title
plt.xlabel('Values (%)', fontweight='bold')
plt.ylabel('Metrics', fontweight='bold')
plt.title('Performance Metrics', fontweight='bold')
plt.tight_layout()

# Show plot
plt.show()