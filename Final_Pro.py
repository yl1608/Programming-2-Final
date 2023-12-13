import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 
import altair as alt
from PIL import Image
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 

# Read data into a dataframe named "s"
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

# Define the target vector (y) and feature set (X)
y = ss['sm_li']  # Target variable
X = ss.drop('sm_li', axis=1)  # Features excluding the target variable

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("X_train shape:", x_train.shape)
print("X_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

log_model = sm.Logit(y_train, x_train).fit() 
log_model.summary()

lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train,y_train)

predicted_labels = lr.predict(x_test)
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy}")

newdata = pd.DataFrame({
    "income": [1,8,8],
    "education":[1,7,1],
    "parent": [0,0,0],
    "married":[0,1,1],
    "female": [0,1,1],
    "age": [12,42,82],
})

newdata["sm_li"] = lr.predict(newdata) #Our new column

newdata

st.markdown("LinkedIn User Prediction App")

"Please select the options that apply to you."
Income = st.selectbox(label="Household Income",
options=("$10,000 to $20,000", 
"$20,000 to $30,000", 
"$40,000 to $50,000", 
"$50,000 to $75,000", 
"$100,000 to $150,000", 
"$150,000+", 
"Don't Know"))

if Income == "$10,000 to $20,000":
    Income =2
elif Income == "$20,000 to $30,000":
    Income =3
elif Income == "$30,000 to $40,000":
    Income =4
else:
    Income=5

Age = st.slider(label="Enter Your Age", 
        min_value=1,
        max_value=100,
        value=50)

education = st.selectbox(label="Highest Level of Education",
options=("Some High School",  
"High School Graduate", 
"Some College, no Degree", 
"Two -year College or University", 
"Bachelor's degree (BA,BS,etc.)", 
"Postgraduate Degree (MA, MS, PhD, MD, ect.)"))

if education == "Some High School":
    education = 1
elif education == "High School Graduate":
    education = 2
else:
    education = 3

Marital = st.selectbox(label="Are you married?",
options=("Yes", 
"No"))
if Marital == "Yes":
    Marital = 1
else:
    Marital = 0

Parent = st.selectbox(label="Are you the parent of a child under the age of 18?",
options=("Yes", 
"No"))

if Parent == "Yes":
    Parent = 1
else:
    Parent = 0

female = st.selectbox(label="Are you a male or female?",
options=("Female", 
"Male"))

if female == "Yes":
    female = 1
else:
    female = 0

st.write("(1 = LinkedIn User, 0 = Not a LinkedIn User)")
