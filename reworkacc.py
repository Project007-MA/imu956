import matplotlib.pyplot as plt
import seaborn as sns

# Set font and general plot appearance
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Data for the graph
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'MCC', 'NPV','FPR',"FNR"]
values = [96.866485, 96.906595, 96.567557, 99.351750, 96.729567, 
    96.155497, 99.364301, 0.648250, 3.432443]  # Example values; replace with your data

# Custom color palette
colors = [
    '#4A90E2',  # Sky blue
    '#6A5ACD',  # Slate blue
    '#8A2BE2',  # Blue violet
    '#9370DB',  # Medium purple
    '#4169E1',  # Royal blue
    '#483D8B',  # Dark slate blue
    '#7B68EE',  # Medium slate blue
    '#5F4B8B',  # Dark orchid
    '#00BFFF',  # Deep sky blue
    '#9B59B6',  # Amethyst purple
]

# Set the style for the plot
# sns.set_theme(style="whitegrid")

# Create the horizontal bar plot
plt.figure(figsize=(12, 6))  # Increase figure size for better bar width
sns.barplot(x=values, y=metrics, palette=colors, orient='h')

# Add labels and title
plt.xlabel("Values")
plt.ylabel("Metrics")
# plt.title("Performance Metrics")

# Set x-axis ticks
plt.xticks()
plt.xlim(0, 100)  # Set x-axis limits for better visibility

# Adjust spacing and display the plot
plt.tight_layout()
plt.show()
