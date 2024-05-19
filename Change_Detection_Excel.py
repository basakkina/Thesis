# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:37:33 2024

@author: Basak
"""

import pandas as pd 
import numpy as np
# Burdaki excel dosyalari Lup tarafindan verilen point.shp´lerden olusturulmustur.
# Birinci deneme Change Detection
Prediction_2018 = pd.read_csv(r"C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Duisburg_2018_points_prediction.csv")
Prediction_2022 = pd.read_csv(r"C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Duisburg_2022_points_prediction.csv")

Prediction_2018_np = np.asarray(Prediction_2018)
Prediction_2022_np = np.asarray(Prediction_2022)

# Get rid of X and Y
Prediction_2018_np_CD = Prediction_2018_np[:, 2].astype('float')
Prediction_2022_np_CD = Prediction_2022_np[:, 2].astype('float')

# mask the arrays with error metrics

mask = Prediction_2018_np_CD > 1.9
mask1 = Prediction_2022_np_CD > 1.9

# error metrics implemented arrays

Prediction_2018_EM = Prediction_2018_np_CD[mask]
Prediction_2022_EM = Prediction_2022_np_CD[mask]

# Keep only the last column
Change_Detection = Prediction_2022_EM = Prediction_2018_EM

No_Change = Change_Detection[(Change_Detection == 0)]
Positive_Change = Change_Detection[(Change_Detection > 0)]
Negative_Change = Change_Detection[(Change_Detection < 0)]
import matplotlib.pyplot as plt

# Histogram
plt.hist(No_Change)
plt.xlabel('No Change')
plt.ylabel('Number')
plt.show()

# %% [2] Daha sonraki adim confusion matrix yapmak, fakat bunu ground truth data´yi kullanarak yapabilir miyim sanmiyorum :D
# Load Ground Truth Data
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')


# Do the Change Detection between Pre2018 and Pre2022
Ground_Truth_Data['Change Detection Prediction'] = Ground_Truth_Data['Pre_2022'].sub(Ground_Truth_Data['Pre_2018'], axis=0)
# Do the Change Detection between Ref2018 and Ref2022
Ground_Truth_Data['Change Detection Reference'] = Ground_Truth_Data['Ref_2022'].sub(Ground_Truth_Data['Ref_2018'], axis=0)
Ground_Truth_Data_np = np.asarray(Ground_Truth_Data)
Data_true = (Ground_Truth_Data_np[:, 5])
Prediction_CD = (Ground_Truth_Data_np[:, 11])
Reference_CD = (Ground_Truth_Data_np[:, 12])
# Making the Numpy arrays suitable for confusion matrix
# First create a copy
Original_Prediction_CD = Prediction_CD.copy()

# Loop through each element in the array
for i in range(len(Prediction_CD)):
    if Prediction_CD[i] < 0:
        Prediction_CD[i] = "Negative_Change"
    elif Prediction_CD[i] > 0:
        Prediction_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Prediction_CD[i] = "No_Change"

# Make the same thing for Reference Change Detection
Original_Reference_CD = Reference_CD.copy()
for i in range(len(Reference_CD)):
    if Reference_CD[i] < 0:
        Reference_CD[i] = "Negative_Change"
    elif Reference_CD[i] > 0:
        Reference_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Reference_CD[i] = "No_Change"
# Confusion matrix

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

# define normalized confusion matrix
Confusion_Matrix_Prediction = confusion_matrix(Data_true, Prediction_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# compute accuracy using sklearn.metrics
accuracy = accuracy_score(Data_true, Prediction_CD)

print("Accuracy:", accuracy)

# Outcome of Accuracy is: 0.4908485856905158

#Precision and Recall

# compute precision using sklearn.metrics
precision = precision_score(Data_true, Prediction_CD, average='macro')

print("Precision(macro):", precision)

#recall 
# compute recall using sklearn.metrics
recall = recall_score(Data_true, Prediction_CD, average='macro')

print("Recall:", recall)

# compute F1 score using sklearn.metrics
F1 = f1_score(Data_true, Prediction_CD, average='macro')

print("F1 Score:", F1)

## Outcome 
# Accuracy: 0.4908485856905158
# Precision(macro): 0.36937992393371366
# Recall: 0.4905058043117745
# F1 Score: 0.4071664640235317

report = classification_report(Data_true, Prediction_CD)

print(report)

# Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Prediction, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
            yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# %% Same Thing For Reference Data

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')
# Do the Change Detection between Ref2018 and Ref2022
Ground_Truth_Data['Change Detection Reference'] = Ground_Truth_Data['Ref_2022'].sub(Ground_Truth_Data['Ref_2018'], axis=0)
Ground_Truth_Data_np = np.asarray(Ground_Truth_Data)
Data_true = (Ground_Truth_Data_np[:, 5])
Reference_CD = (Ground_Truth_Data_np[:, 11])

# Making the Numpy arrays suitable for confusion matrix
# First create a copy
Original_Reference_CD = Reference_CD.copy()

# Loop through each element in the array
# Make the same thing for Reference Change Detection

Original_Reference_CD = Reference_CD.copy()
for i in range(len(Reference_CD)):
    if Reference_CD[i] < 0:
        Reference_CD[i] = "Negative_Change"
    elif Reference_CD[i] > 0:
        Reference_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Reference_CD[i] = "No_Change"

# Confusion matrix

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
# define normalized confusion matrix
Confusion_Matrix_Reference = confusion_matrix(Data_true, Reference_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Reference, annot=True, fmt="d", cmap="Blues", 
           xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
           yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Reference Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# compute accuracy using sklearn.metrics
accuracy = accuracy_score(Data_true, Reference_CD)

print("Accuracy:", accuracy)

# Outcome of Accuracy is: 0.4908485856905158

#Precision and Recall

# compute precision using sklearn.metrics
precision = precision_score(Data_true, Reference_CD, average='macro')

print("Reference(macro):", precision)

#recall 
# compute recall using sklearn.metrics
recall = recall_score(Data_true, Reference_CD, average='macro')

print("Recall:", recall)

# compute F1 score using sklearn.metrics
F1 = f1_score(Data_true, Reference_CD, average='macro')

print("F1 Score:", F1)

report = classification_report(Data_true, Reference_CD)

print(report)

# %% Now implementing error metrics on the CD
# Load Ground Truth Data, Prediction Data and Reference Data
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')

# Do the Change Detection between Pre2018 and Pre2022
Ground_Truth_Data['Change Detection Prediction'] = Ground_Truth_Data['Pre_2022'].sub(Ground_Truth_Data['Pre_2018'], axis=0)
# Do the Change Detection between Ref2018 and Ref2022
Ground_Truth_Data['Change Detection Reference'] = Ground_Truth_Data['Ref_2022'].sub(Ground_Truth_Data['Ref_2018'], axis=0)
Ground_Truth_Data_np = np.asarray(Ground_Truth_Data)
Data_true = (Ground_Truth_Data_np[:, 5])
Prediction_CD = (Ground_Truth_Data_np[:, 11])
Reference_CD = (Ground_Truth_Data_np[:, 12])

# Making the Numpy arrays suitable for confusion matrix
# First create a copy
Original_Prediction_CD = Prediction_CD.copy()

# Loop through each element in the array
for i in range(len(Prediction_CD)):
    if  Prediction_CD[i] < -0.35:
        Prediction_CD[i] = "Negative_Change"
    elif Prediction_CD[i] > 0.35:
        Prediction_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Prediction_CD[i] = "No_Change"

# Make the same thing for Reference Change Detection
Original_Reference_CD = Reference_CD.copy()
for i in range(len(Reference_CD)):
    if Reference_CD[i] < -0.35:
        Reference_CD[i] = "Negative_Change"
    elif Reference_CD[i] > 0.35:
        Reference_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Reference_CD[i] = "No_Change"
        
# Confusion matrix

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
# define normalized confusion matrix
Confusion_Matrix_Prediction = confusion_matrix(Data_true, Prediction_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Prediction, annot=True, fmt="d", cmap="Blues", 
           xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
           yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Prediction Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

Confusion_Matrix_Reference = confusion_matrix(Data_true, Reference_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Reference, annot=True, fmt="d", cmap="Blues", 
           xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
           yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Reference Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# compute accuracy using sklearn.metrics
accuracy = accuracy_score(Data_true, Reference_CD)

print("Accuracy:", accuracy)

# Outcome of Accuracy is: 0.4908485856905158

#Precision and Recall

# compute precision using sklearn.metrics
precision = precision_score(Data_true, Reference_CD, average='macro')

print("Reference(macro):", precision)

#recall 
# compute recall using sklearn.metrics
recall = recall_score(Data_true, Reference_CD, average='macro')


print("Recall:", recall)

# compute F1 score using sklearn.metrics
F1 = f1_score(Data_true, Reference_CD, average='macro')

print("F1 Score:", F1)

report = classification_report(Data_true, Prediction_CD)

print(report)

