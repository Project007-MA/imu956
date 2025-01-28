import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
def plot1():
    data = {
        'Classifier': ['YOLO3', 'YOLO4 ', 'YOLO5', 'YOLO7', 'Proposed'],
        'Precision': [0.654, 0.681, 0.702, 0.741, 0.9897],
        'Recall': [0.694, 0.722, 0.702, 0.771, 0.993],
        'F1score': [0.663, 0.681, 0.693, 0.792, 0.99],
        'mAP': [0.721, 0.754, 0.734, 0.791, 0.984]
    }


    df = pd.DataFrame(data)


    df_melted = df.melt(id_vars="Classifier", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))

    sns.barplot(x='Classifier', y='Score', hue='Metric', data=df_melted,palette="twilight_shifted")

    plt.ylabel('Values')
    plt.xlabel('Models')
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("Results/c_plot1.png",dpi=1000)
    plt.show()


def plot2():

    # Data
    data = {
        'Algorithms': ['Random Forest', 'Linear Model', 'SVR', 'Proposed'],
        'MAE': [0.08, 0.09, 0.10, 0.03],
        'rRMSE': [0.37, 0.38, 0.40, 0.12]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set the figure size
    plt.figure(figsize=(12, 7))

    # Plot MAE as a line
    sns.lineplot(x='Algorithms', y='MAE', data=df, marker='>', label='MAE', linewidth=2)

    # Plot rRMSE as a line
    sns.lineplot(x='Algorithms', y='rRMSE', data=df, marker='D', label='rRMSE', linewidth=2)

    plt.ylabel('Values')
    plt.xlabel('Methods')

    # Set x-tick labels with a higher font size and bold text
    plt.xticks()
    plt.yticks()

    # Show legend
    plt.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.savefig("Results/c_plot2.png", dpi=1000)
    plt.show()


def plot3():

    # Data
    data = {
        'Models': ['CapsNet', 'DensNet 121', 'Leaky-SENet', 'Proposed'],
        'Accuracy': [95, 95, 92, 99.1],
        'Precision': [97, 94.5, 90, 98.97],
        'Recall': [95, 94.8, 88, 99.3],
        'F1score': [96, 94.5, 89, 99.1]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set the figure size
    plt.figure(figsize=(12, 7))

    # Melt the DataFrame to plot multiple metrics as bars
    df_melted = df.melt(id_vars="Models", var_name="Metric", value_name="Value")

    # Create the bar plot
    sns.barplot(x='Models', y='Value', hue='Metric', data=df_melted, palette='twilight_shifted')

    # Add title and labels

    plt.xlabel('Methods')
    plt.ylabel('Values')
    plt.xticks()
    plt.yticks()


    # Show the legend with increased font size
    plt.legend(loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.savefig("Results/c_plot3.png", dpi=1000)
    plt.show()

def plot4():
    import matplotlib.pyplot as plt
    import pandas as pd

    # Data
    data = {
        'Class': ['Homogenous', 'Speckled', 'Nucleolar', 'Centromere', 'Golgi', 'Nuclear Membrane'],
        'Samples': [2494, 2831, 2598, 2741, 724, 2208]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set the figure size
    plt.figure(figsize=(12, 6))

    # Create the lollipop plot
    plt.stem(df['Class'], df['Samples'], linefmt='r-', markerfmt='bo', basefmt=' ')

    # Add title and labels
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks()
    plt.yticks()

    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.savefig("Results/c_plot4.png", dpi=1000)
    plt.show()
#plot1()
#plot2()
#plot3()
plot4()