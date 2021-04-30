# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:33:58 2021

@author: Ozan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


data = pd.read_csv('C:/Users/Ozan/Desktop/diabetes.csv')


print(data.isnull().sum())
print(data.isna().sum())


col = data.columns


diabetes_copy  = data.copy(deep = True)

diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin',
               'BMI']] = diabetes_copy[['Glucose','BloodPressure',
                                             'SkinThickness','Insulin',
                                             'BMI']].replace(0,np.NaN)





# diabetes_copy.hist(figsize=(25,25))
# plt.show()

# figure = plt.figure(figsize=(25,25))
# sns.pairplot(diabetes_copy, hue='Outcome')
# plt.show()


corr_Pearson = diabetes_copy.corr(method='pearson')

figure = plt.figure(figsize=(25,15))
sns.heatmap(corr_Pearson,vmin=-1,vmax=+1,cmap='Blues',annot=True, 
            linewidths=1,linecolor = 'white')
plt.title('Pearson Correlation')
# plt.savefig('/cor.png')
plt.show()









































