import pandas as pd

df = pd.read_csv("Data/EV_Battery_Dataset.csv")
print("---   Dataset Loaded...   ---".center(125))
print(df)

print("---   Dataset Information   ---".center(125))
print(df.info())

print("---   Dataset Default Top Five Data   ---".center(125))
print(df.head())    

print("---   Dataset Default Bottom Five Data   ---".center(125))
print(df.tail())

print("---   Dataset Data Types   ---".center(125))
print(df.dtypes)

print("---   Dataset Total Rows and Columns   ---".center(125))
print(df.shape)

print("---   Dataset Numeric (Quantitative Data)   ---".center(125))
qantitive = df.select_dtypes(include=["int64","float64"])
print(qantitive)

print("---   Dataset Categorical (Qualitative Data)   ---".center(125))
qulatative = df.select_dtypes(include=["object","category"])
print(qulatative)

print("---   Dataset Null Values   ---".center(125))
print(df.isnull().sum())

print("---   Dataset Unique Values   ---".center(125))
print(df.nunique())

print("---   Dataset Duplicate Values   ---".center(125))
print(df.duplicated().sum())

print("---   Dataset Statistical Summary   ---".center(125))
print(df.describe())