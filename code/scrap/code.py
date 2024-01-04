import pandas as pd
import matplotlib.pyplot as plt

excel_file = "train8.xlsx"
# 35 healthy persons, with 50 to 60 PPG signal for each one. 
# Each PPG signal contains 300 samples (6 seconds recording) with 50 sample/second sampling rate.
dataset = pd.read_excel(excel_file)
print(dataset) #reading the excel file

dataset.head() #displays the first few rows of a DataFrame - [304 rows x 1374 columns]

# Standardize the dataset; This is very important before you apply PCA

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
