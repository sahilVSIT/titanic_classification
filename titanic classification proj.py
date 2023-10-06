import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv("titanic_ds.csv")

print("First 5 rows of the dataset:")
print(titanic_data.head())

print("Number of rows and columns:")
print(titanic_data.shape)

print("Missing values per column:")
print(titanic_data.isnull().sum())

# Data Preprocessing (Removed)
# -----------------------------

print("Statistical summary of the dataset:")
print(titanic_data.describe())

print("Counts of survival (0 = No, 1 = Yes):")
print(titanic_data['Survived'].value_counts())

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=titanic_data)

plt.subplot(2, 2, 2)
sns.countplot(x='Sex_male', data=titanic_data)

plt.subplot(2, 2, 3)
sns.countplot(x='Sex_male', hue='Survived', data=titanic_data)

plt.subplot(2, 2, 4)
sns.countplot(x='Sex_female', hue='Survived', data=titanic_data)

plt.subplot(2, 2, 4)
sns.countplot(x='Pclass', data=titanic_data)

X = titanic_data.drop(columns=['PassengerId', 'Name', 'Survived'], axis=1)
Y = titanic_data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print("Training accuracy:", training_accuracy)

Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print("Test accuracy:", test_accuracy)
