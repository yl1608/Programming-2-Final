import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Read data into a dataframe named "s"

#p = os.getcwd() + '\social_media_usage.csv'
#s = pd.read_csv(p)
csv_file_path = "social_media_usage.csv"
s = pd.read_csv(csv_file_path)
# Check the dimensions of the dataframe
print(s.shape)

s.head()

# Define the clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create a toy dataframe
data = {'Column1': [1, 0, 1], 'Column2': [0, 1, 1]}
toy_df = pd.DataFrame(data)

# Apply the clean_sm function to each element in the dataframe
toy_df_cleaned = toy_df.applymap(clean_sm)

# Display the original and cleaned dataframes
print("Original DataFrame:")
print(toy_df)

print("\nCleaned DataFrame:")
print(toy_df_cleaned)

# Create a toy dataframe "ss"
ss0 = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    "income":np.where(s["income"]>9, np.nan, s["income"]),
    "education":np.where(s["educ2"] >9,np.nan, s["educ2"]),
    "parent":np.where(s["par"]==1,1,0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"]>98, np.nan, s["age"]),})

ss=ss0.dropna()
ss.columns
ss.head()


# Drop missing values
ss = ss.dropna()

ss.describe()

# Define the target vector (y) and feature set (X)
y = ss['sm_li']  # Target variable
X = ss.drop('sm_li', axis=1)  # Features excluding the target variable

# Display the target vector and feature set
print("Target Vector (y):")
print(y)

print("\nFeature Set (X):")
print(X)

from sklearn.model_selection import train_test_split

# Define the target vector (y) and feature set (X)
y = ss['sm_li']  # Target variable
X = ss.drop('sm_li', axis=1)  # Features excluding the target variable

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import statsmodels.api as sm 

# Display the shapes of the resulting sets
print("X_train shape:", x_train.shape)
print("X_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

log_model = sm.Logit(y_train, x_train).fit() 
log_model.summary()

log_model_sklearn = LogisticRegression(class_weight="balanced")
log_model_sklearn.fit(x_train,y_train)