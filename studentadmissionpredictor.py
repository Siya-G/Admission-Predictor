
import pandas as pd
import numpy as np

df = pd.read_csv('Admission_Predict.csv')

df.shape

df.isnull().sum()

df.drop(columns='Serial No.', inplace=True)

df['Chance of Admit '] = (df['Chance of Admit '] >= 0.75).astype(int)

df[df['GRE Score'] < 303]['Chance of Admit '].value_counts(normalize=True)

df[df['CGPA'] < 8.0]['Chance of Admit '].value_counts(normalize=True)

df[(df['GRE Score']) < 340 & (df['CGPA'] < 8.0)]['Chance of Admit '].value_counts(normalize=True)

df[df['TOEFL Score'] < 98]['Chance of Admit '].value_counts(normalize=True)

df[df['University Rating'] < 2]['Chance of Admit '].value_counts(normalize=True)

df[df['GRE Score'] < 310]['Chance of Admit '].value_counts(normalize=True)

df[df['SOP'] < 2]['Chance of Admit '].value_counts(normalize=True)

df[df['LOR '] < 2]['Chance of Admit '].value_counts(normalize=True)

df[df['Research'] == 0]['Chance of Admit '].value_counts(normalize=True)

df['low_gre'] = (df['GRE Score'] < 303).astype(int)

df['low_cgpa'] = (df['CGPA'] < 8.0).astype(int)

df['low_toefl'] = (df['TOEFL Score'] < 98).astype(int)

df['weak_sop'] = (df['SOP'] < 2).astype(int)

import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(x='Chance of Admit ', y='GRE Score', data=df)
plt.title("GRE Scores by Admission Decision")
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')

df['academic score'] = df['GRE Score'] + df['TOEFL Score']
df.drop(columns=['GRE Score', 'TOEFL Score'], inplace=True)

df.columns

from sklearn.model_selection import train_test_split

features = ['University Rating', 'SOP', 'LOR ', 'CGPA', 'Research',
             'low_cgpa', 'weak_sop', 'academic score']
X = df[features]
y = df['Chance of Admit ']

df['Chance of Admit '][1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train_scaled, y_train)

import pickle

with open("admission_model.pkl", "wb") as f:
  pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
  pickle.dump(scaler, f)

from google.colab import files

files.download("admission_model.pkl")
files.download("scaler.pkl")

y_pred = model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Model score:", model.score(X_test_scaled, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

