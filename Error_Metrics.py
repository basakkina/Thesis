# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:14:35 2024

@author: Basak
"""

# Error metrics
# 1. Load the data
import pandas as pd 
import numpy as np
Prediction_2018 = pd.read_csv(r"C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Duisburg_2018_points_prediction.csv")
#Reference_2018 = pd.read_excel(r"C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Duisburg_2018_points_reference.xlsx")
#Prediction_2022 = pd.read_excel(r"C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Duisburg_2022_points_prediction.xlsx")
#Reference_2022 = pd.read_excel(r"C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Duisburg_2022_points_reference.xlsx")


# pandas to numpy
Prediction_2018_np = np.asarray(Prediction_2018)
#Reference_2018_np = np.asarray(Reference_2018)

# keep the only 3rd coloum
Prediction_2018_np = Prediction_2018_np[:, 2].astype('float')


#stop
#Comparison_MAE = np.sum(np.abs(data_prediction_numpy - data_reference_numpy)) / number_of_values
#df['Comparison_MAE'] = np.where((Reference_2018['X'] == Prediction_2018['X']) & (Reference_2018['Y'] == Prediction_2018['Y']), Prediction_2018['Pre_181'])

                            

                 


