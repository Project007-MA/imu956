import matplotlib.pyplot as plt

# Set global font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data for accuracy graph
epochs = list(range(1, 16))  # X-axis (e.g., epochs from 1 to 15)
accuracy = [0.86, 0.89, 0.9, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.942, 0.943, 0.944, 0.9445, 0.945]
dashed_accuracy = [0.88, 0.89, 0.903, 0.909, 0.91, 0.912, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.942, 0.943, 0.944]

# Data for loss graph
loss = [0.4, 0.35, 0.32, 0.28, 0.25, 0.22, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1]
dashed_loss = [0.42, 0.38, 0.34, 0.3, 0.27, 0.24, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11]

# Create the figure and subplots
plt.figure(figsize=(12, 7))

# Subplot 1: Accuracy graph
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
plt.plot(epochs, accuracy, label="Training", linestyle="-", marker="o", linewidth=2.5)
plt.plot(epochs, dashed_accuracy, label="Validation", linestyle="--", linewidth=2.5)
plt.xlabel("Epochs")
plt.ylabel("Values")
plt.title("Accuracy")
plt.legend()

# Subplot 2: Loss graph
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
plt.plot(epochs, loss, label="Training", linestyle="-", marker="o", linewidth=2.5)
plt.plot(epochs, dashed_loss, label="Validation", linestyle="--", linewidth=2.5)
plt.xlabel("Epochs")
plt.ylabel("Values")
plt.title("Loss")
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
