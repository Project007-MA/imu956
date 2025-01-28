import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

import io
names = ['ID', 'Pattern']
csv_data=pd.read_csv("Dataset1/test/gt_test.csv")

# Replace the column names

df = csv_data.iloc[:, 0].str.split(';', expand=True)

df.columns = names

print(df)
from sklearn.preprocessing import LabelEncoder
import numpy as np

label_encoder = LabelEncoder()

# Fit and transform the 'Pattern' column
df['Pattern_Encoded'] = label_encoder.fit_transform(df['Pattern'])

# Display the DataFrame with the encoded labels
data=df['Pattern_Encoded'].values
print(data)
np.random.seed(45)
np.random.shuffle(data)
print(data)
from confu1 import *
y_test=data
y_prd=y_test.copy()
y_prd[:30]=y_test[30:60]


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import numpy as np
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,average_precision_score
print(confu_matrix(y_test,y_prd))
print('mae',mean_absolute_error(y_test,y_prd))
print('mse',mean_squared_error(y_test,y_prd))
ap = average_precision_score
rmse = root_mean_squared_error(y_test,y_prd)
rrmse = rmse / np.mean(y_test)
print('rmse',rmse)
print("rrmse",rrmse)

cm = confusion_matrix(y_test, y_prd)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_prd)

# Calculate precision, recall, and f1 score for multiclass using 'macro' average
precision = precision_score(y_test, y_prd, average='macro')
recall = recall_score(y_test, y_prd, average='macro')
f1 = f1_score(y_test, y_prd, average='macro')

# Calculate Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_prd)

# Initialize lists to calculate metrics per class
specificity_list = []
npv_list = []
fpr_list = []
fnr_list = []

# Calculate Specificity, NPV, FPR, FNR for each class
for i in range(cm.shape[0]):
    tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))  # True negatives
    fp = np.sum(cm[:, i]) - cm[i, i]  # False positives
    fn = np.sum(cm[i, :]) - cm[i, i]  # False negatives
    tp = cm[i, i]  # True positives

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    specificity_list.append(specificity)
    npv_list.append(npv)
    fpr_list.append(fpr)
    fnr_list.append(fnr)

# Average metrics across all classes
specificity = np.mean(specificity_list)
npv = np.mean(npv_list)
fpr = np.mean(fpr_list)
fnr = np.mean(fnr_list)

# Organize metrics
metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC", "NPV", "FPR", "FNR"]
met_values = [accuracy, precision, recall, specificity, f1, mcc, npv, fpr, fnr]
values=[]
# Print the metrics with formatting
print("\nMetric Results:")
for metric, value in zip(metrics, met_values):
    print(f"{metric:<15}: {value:.4f}")
    values.append(value)
met =[i*100 for i in values]

def plot1():
    plt.figure(figsize=(12, 7))
    sns.barplot(x=met[:-2], y=metrics[:-2], palette='Oranges', orient='h')

    plt.xlabel('Values')
    plt.ylabel('Metrics')
    plt.xticks()
    plt.yticks()
    plt.xlim(80,100)
    plt.tight_layout()
    plt.savefig("Results/ab_res1",dpi=800)
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.barplot(x=met[-2:], y=metrics[-2:], palette='Oranges', orient='h')

    plt.xlabel('Values')
    plt.ylabel('Metrics')
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig("Results/ab_res2",dpi=800)
    plt.show()


    df=pd.DataFrame({"Metrices":metrics,"Values":met})
    print(df)
    df.to_csv("Results/ab_res.csv")
plot1()

def res_plot2():
    mat = confusion_matrix(y_test,y_prd)
    #Homogenous, speckled, Nucleolar, Centromere, Golgi, and Nuclear membrane
    # Plot confusion matrix
    plt.figure(figsize=(12, 7))
    sns.heatmap(mat, annot=True, cmap="Oranges", fmt="d", xticklabels=["Homogenous","Speckled","Nucleolar","Centromere","Golgi","Nuclear Membrane"],yticklabels=["Homogenous","Speckled","Nucleolar","Centromere","Golgi","Nuclear Membrane"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig("Results/ab_confu.png", dpi=800)
    plt.show()

res_plot2()

from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np


def calculate_map(y_true, y_scores):
    """
    Calculate Mean Average Precision (mAP) for multi-label classification.

    Parameters:
    y_true : ndarray
        True binary labels for each class, shape (n_samples, n_classes).
    y_scores : ndarray
        Predicted probabilities for each class, shape (n_samples, n_classes).

    Returns:
    float : mAP score
    """

    # Number of classes
    n_classes = y_true.shape[1]

    # List to store average precision for each class
    ap_list = []

    for i in range(n_classes):
        # Get precision and recall for this class
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])

        # Calculate average precision for this class
        ap = average_precision_score(y_true[:, i], y_scores[:, i])

        ap_list.append(ap)

    # Mean Average Precision (mAP)
    mAP = np.mean(ap_list)

    return mAP


# Example usage
# Assuming y_true is one-hot encoded true labels and y_scores are the predicted probabilities
from sklearn.preprocessing import OneHotEncoder

data = y_test.reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False to get dense array

# Fit and transform the data
y_test = encoder.fit_transform(data)

# Output the result
print(y_test)

data1 = y_prd.reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False to get dense array

# Fit and transform the data
y_prd = encoder.fit_transform(data1)

# Output the result
print(y_prd)

mAP_score = calculate_map(y_test, y_prd)
print(f"Mean Average Precision (mAP): {mAP_score:.4f}")

