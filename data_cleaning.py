import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

print("task 1: Import the dataset and explore basic information")

df=pd.read_csv('Titanic-Dataset.csv')
print("Initial DataFrame:")
print(df.head())
print("DataFrame Shape:", df.shape)
print("Dataset info:")
print(df.info())
print("Null values in each column:")
print(df.isnull().sum())
print("Descriptive statistics:")
print(df.describe())

print("task 2: Handle missing values using mean/median/imputation.")

print("Missing values:")
print(df.isnull().sum())

# Fill missing 'Age' with median
print("Filling missing 'Age' values with median...")
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode
print("Filling missing 'Embarked' values with mode...")
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to high number of missing values
print("Dropping 'Cabin' column due to high number of missing values...")
df.drop(columns=['Cabin'], inplace=True)
print("Missing values after handling:")
print(df.isnull().sum())

print("task 3: Convert categorical variables to numerical using encoding techniques.")

df['Sex']=df["Sex"].map({'male':0,'female':1})
print("sex colum encoded")
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print("Embarked column encoded using one-hot encoding")
print("columns after encoding:")
print(df.columns)

print("task 4: Normalize the numerical features using standardization.")
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Numerical features after standardization:")
print(df[numeric_cols].head())


print("task 5: Visualize outliers using boxplots and remove them.")
# Boxplot for visualizing outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.show()

# Remove outliers using IQR method
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Filter out rows with any feature beyond 1.5*IQR
df_clean = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Shape after removing outliers:", df_clean.shape)
print("Cleaned DataFrame after outlier removal:")
print(df_clean.head())
# Save the cleaned DataFrame after outlier removal
df_clean.to_csv('Titanic-Dataset-Cleaned.csv', index=False)
print("Cleaned dataset after outlier removal saved to 'Titanic-Dataset-Cleaned.csv'.")


