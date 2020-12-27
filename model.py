import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import pickle
import os

#Load Data
df  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

#Preprocessing
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)
df.dropna(inplace=True)
df = df.astype({'Credit_History': 'object'})

df.drop(['Unnamed: 0','Loan_ID','LoanAmount','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term'],axis = 1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df.drop(['Dependents_1','Dependents_3+','Education_Not Graduate','Self_Employed_Yes','Property_Area_Urban'], axis = 1, inplace=True)

#Split data
X = df.drop(['Loan_Status'], axis =1)
y = df['Loan_Status']
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=29, stratify = y)
X_train.head()

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

#Test model
result = f1_score(y_test, model.predict(X_test))
print(result)

#Save model
filename = 'Loan_mdl.pkl'
pickle.dump(model, open(filename, 'wb'))

import json
columns = {
    'data_columns' : [col.lower() for col in X_train.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))

