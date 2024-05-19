# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:22:27 2024

@author: Basak
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

Ground_Truth_Data = pd.read_csv(r'C:\\Users\\Basak\\Documents\\Masterarbeit\\Data\\Excel_08.02\\Ground_Truth_With_Rasters.csv')


# Do the Change Detection between Pre2018 and Pre2022
Ground_Truth_Data['Change Detection Prediction'] = Ground_Truth_Data['Pre_2022'].sub(Ground_Truth_Data['Pre_2018'], axis=0)
# Do the Change Detection between Ref2018 and Ref2022
Ground_Truth_Data['Change Detection Reference'] = Ground_Truth_Data['Ref_2022'].sub(Ground_Truth_Data['Ref_2018'], axis=0)
Change_Detection = Ground_Truth_Data[['Change Detection Prediction','Change Detection Reference']].copy()

Change_Detection.plot.scatter(x="Change Detection Prediction", y="Change Detection Reference", colormap="viridis", alpha=0.5)
plt.show()

boxplot = Change_Detection.boxplot(column=['Change Detection Prediction', 'Change Detection Reference'])  

pd.plotting.scatter_matrix(Change_Detection, alpha=0.2)

plot = Change_Detection.plot(title="Change_Detection Plot")

# pandas to numpy
#Ground_Truth_Data_np = np.asarray(Ground_Truth_Data)
#Prediction_CD = (Ground_Truth_Data_np[:, 11])
#Reference_CD = (Ground_Truth_Data_np[:, 12])

#plt.hist(Prediction_CD)
#plt.xlabel('Prediction')
#plt.ylabel('Changes')
#plt.title('Prediction Change Detection Histogram')
#plt.show()

#plt.hist(Reference_CD)
#plt.xlabel('Reference')
#plt.ylabel('Changes')
#plt.title('Reference Change Detection Histogram')
#plt.show()