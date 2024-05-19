# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:40:27 2024

@author: Basak
"""
# 1. Ground Truth Data is selected
# 2. Implement Change Detection without Noise Elimination aka Error Metrics
# 3. Implement Confusion Matrix to see how accurate the "raw" model predicts changes 
# 4. Implement Noise Elimination aka Error Metrics 
# 5. Implement Confusion Matrix on "changed" model 

# %% [1] Ground Truth Data is selected
# Load Ground Truth Data

import pandas as pd 
import numpy as np
Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')

# [2] Implement Change Detection without Noise Elimination aka Error Metrics

#Do the Change Detection between Pre2018 and Pre2022
Ground_Truth_Data['Change Detection Prediction'] = Ground_Truth_Data['Pre_2022'].sub(Ground_Truth_Data['Pre_2018'], axis=0)
# Do the Change Detection between Ref2018 and Ref2022
Ground_Truth_Data['Change Detection Reference'] = Ground_Truth_Data['Ref_2022'].sub(Ground_Truth_Data['Ref_2018'], axis=0)

# [3] Implement Confusion Matrix to see how accurate the "raw" model predicts changes 

# Panda to Numpy
Ground_Truth_Data_np = np.asarray(Ground_Truth_Data)
# Preparing the data for Confusion Matrix
# Extracting "Data True"
Data_true = (Ground_Truth_Data_np[:, 5])
# Extracting Change Detection Prediction and Reference
Prediction_CD = (Ground_Truth_Data_np[:, 11])
Reference_CD = (Ground_Truth_Data_np[:, 12])
# As in the "Data True" the values are string, first we need to change float values of Change Detection Prediction and Reference to string as well

# First start with Change Detection Prediction

for i in range(len(Prediction_CD)):
    if Prediction_CD[i] < 0:
        Prediction_CD[i] = "Negative_Change"
    elif Prediction_CD[i] > 0:
        Prediction_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Prediction_CD[i] = "No_Change"

# Now Change Detection Reference

for i in range(len(Reference_CD)):
    if Reference_CD[i] < 0:
        Reference_CD[i] = "Negative_Change"
    elif Reference_CD[i] > 0:
        Reference_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Reference_CD[i] = "No_Change"
        
# as the datas are ready now we can implement confusion matrix
# first we need to load necessary libraries

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# define normalized confusion matrix for Prediction

Confusion_Matrix_Prediction = confusion_matrix(Data_true, Prediction_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# compute accuracy using sklearn.metrics
accuracy_P = accuracy_score(Data_true, Prediction_CD)
precision_P = precision_score(Data_true, Prediction_CD, average='macro')
recall_P = recall_score(Data_true, Prediction_CD, average='macro')
F1_P = f1_score(Data_true, Prediction_CD, average='macro')
Report_P = classification_report(Data_true, Prediction_CD)
#print(Report_P)

# Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Prediction, annot=True, fmt="d", cmap="Blues", 
               xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
               yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# For Reference
Confusion_Matrix_Reference = confusion_matrix(Data_true, Reference_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])
accuracy_R = accuracy_score(Data_true, Reference_CD)
precision_R = precision_score(Data_true, Reference_CD, average='macro')
recall_R = recall_score(Data_true, Reference_CD, average='macro')
F1_R = f1_score(Data_true, Reference_CD, average='macro')
Report_R = classification_report(Data_true, Reference_CD)
#print(Report_R)

# Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Reference, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
            yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Reference Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
#
# %% [4] Implement Noise Elimination aka Error Metrics
# Change Detection Error Metrics Implemented
import pandas as pd 
import numpy as np

Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')
#Error Metrics MAE implemeted
Error_Metrics_MAE = Ground_Truth_Data.loc[(Ground_Truth_Data['Pre_2022'] > 1.86) & (Ground_Truth_Data['Pre_2018'] > 1.92)]
# Now Do the Change Detection between Pre2018 and Pre2022
Error_Metrics_MAE['Change Detection Prediction'] = Error_Metrics_MAE['Pre_2022'].sub(Ground_Truth_Data['Pre_2018'], axis=0)

# Implement Confusion Matrix to see if there is improvement
# Panda to Numpy
Error_Metrics_MAE_np = np.asarray(Error_Metrics_MAE)
# Preparing the data for Confusion Matrix
# Extracting "Data True"
Data_true = (Error_Metrics_MAE_np[:, 5])
# Extracting Change Detection Prediction and Reference
Prediction_CD = (Error_Metrics_MAE_np[:, 11])
# As in the "Data True" the values are string, first we need to change float values of Change Detection Prediction and Reference to string as well
# First start with Change Detection Prediction

for i in range(len(Prediction_CD)):
    if Prediction_CD[i] < 0:
        Prediction_CD[i] = "Negative_Change"
    elif Prediction_CD[i] > 0:
        Prediction_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Prediction_CD[i] = "No_Change"
        
# as the datas are ready now we can implement confusion matrix
# first we need to load necessary libraries

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# define normalized confusion matrix for Prediction

Confusion_Matrix_Prediction = confusion_matrix(Data_true, Prediction_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# compute accuracy using sklearn.metrics
accuracy_P = accuracy_score(Data_true, Prediction_CD)
precision_P = precision_score(Data_true, Prediction_CD, average='macro')
recall_P = recall_score(Data_true, Prediction_CD, average='macro')
F1_P = f1_score(Data_true, Prediction_CD, average='macro')
Report_P = classification_report(Data_true, Prediction_CD)
#print(Report_P)

#Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Prediction, annot=True, fmt="d", cmap="Blues", 
               xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
               yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# %% [5] Error Metrics MSE
# Change Detection Error Metrics Implemented
import pandas as pd 
import numpy as np

Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')
#Error Metrics MAE implemeted
Error_Metrics_MAE = Ground_Truth_Data.loc[(Ground_Truth_Data['Pre_2022'] > 7.94) & (Ground_Truth_Data['Pre_2018'] > 8.32)]
# Now Do the Change Detection between Pre2018 and Pre2022
Error_Metrics_MAE['Change Detection Prediction'] = Error_Metrics_MAE['Pre_2022'].sub(Ground_Truth_Data['Pre_2018'], axis=0)

# Implement Confusion Matrix to see if there is improvement
# Panda to Numpy
Error_Metrics_MAE_np = np.asarray(Error_Metrics_MAE)
# Preparing the data for Confusion Matrix
# Extracting "Data True"
Data_true = (Error_Metrics_MAE_np[:, 5])
# Extracting Change Detection Prediction and Reference
Prediction_CD = (Error_Metrics_MAE_np[:, 11])
# As in the "Data True" the values are string, first we need to change float values of Change Detection Prediction and Reference to string as well
# First start with Change Detection Prediction

for i in range(len(Prediction_CD)):
    if Prediction_CD[i] < 0:
        Prediction_CD[i] = "Negative_Change"
    elif Prediction_CD[i] > 0:
        Prediction_CD[i] = "Positive_Change"
    else:  # If the value is exactly 0
        Prediction_CD[i] = "No_Change"
        
# as the datas are ready now we can implement confusion matrix
# first we need to load necessary libraries

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# define normalized confusion matrix for Prediction

Confusion_Matrix_Prediction = confusion_matrix(Data_true, Prediction_CD, labels=["No_Change", "Negative_Change", "Positive_Change"])

# compute accuracy using sklearn.metrics
accuracy_P = accuracy_score(Data_true, Prediction_CD)
precision_P = precision_score(Data_true, Prediction_CD, average='macro')
recall_P = recall_score(Data_true, Prediction_CD, average='macro')
F1_P = f1_score(Data_true, Prediction_CD, average='macro')
Report_P = classification_report(Data_true, Prediction_CD)
print(Report_P)

#Show it in heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_Matrix_Prediction, annot=True, fmt="d", cmap="Blues", 
               xticklabels=['No_Change', 'Negative_Change', 'Positive_Change'], 
               yticklabels=['No_Change', 'Negative_Change', 'Positive_Change'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
