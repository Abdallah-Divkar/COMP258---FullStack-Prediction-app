import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = os.path.dirname(os.path.abspath(__file__))
fullpath = r"Student data.csv"
filename = 'Student data.csv'

fullpath = os.path.join(path,filename)
df = pd.read_csv(fullpath,sep=',', skiprows=24, header=None)

df.columns = [
    "first_term_gpa",
    "second_term_gpa",
    "first_language",
    "funding",
    "school",
    "fasttrack",
    "coop",
    "residency",
    "gender",
    "previous_education",
    "age_group",
    "high_school_avg",
    "math_score",
    "english_grade",
    "first_year_persistence"
]
X = df.drop("first_year_persistence", axis=1)
y = df["first_year_persistence"]
numeric_cols = [
    "first_term_gpa",
    "second_term_gpa",
    "high_school_avg",
    "math_score"
]
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

#Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Missing Indicators and filling
for col in numeric_cols:
    X_train[col + "_missing"] = X_train[col].isnull().astype(int)
    X_test[col + "_missing"] = X_test[col].isnull().astype(int)
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
for col in numeric_cols:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

#One hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# align columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("************************************")
print(pd.DataFrame(X_train).isnull().sum().sum())
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

'''numeric_cols = [
    "first_term_gpa",
    "second_term_gpa",
    "high_school_avg",
    "math_score"
]'''
'''for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')'''
'''print("************************************")
print(df.shape)        # should be (1437, 15)
print(df.head())
print(df.info())
print(df.dtypes)
print("************************************")
print(df.isnull().sum())'''
#Dropping columns
#df = df.drop(columns=["school"])

#Missiing values handling
'''for col in ["high_school_avg", "math_score"]:
    df[col + "_missing"] = df[col].isnull().astype(int)'''

'''for col in ["first_term_gpa", "second_term_gpa", "high_school_avg", "math_score"]:
    df[col] = df[col].fillna(df[col].median())

print(df[["high_school_avg_missing", "math_score_missing"]].sum())
print(df.isnull().sum())'''

# Save cleaned dataset
'''output_path = os.path.join(path, "cleaned_student_data.csv")
df.to_csv(output_path, index=False)

print("Cleaned dataset saved to:", output_path)'''

'''fullpath = os.path.join(path, "cleaned_student_data.csv")
df = pd.read_csv(fullpath)

print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)'''

#One hot encoding
#df = pd.get_dummies(df, drop_first=True)

'''X = df.drop("first_year_persistence", axis=1)
y = df["first_year_persistence"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)'''

'''for col in ["first_term_gpa", "second_term_gpa", "high_school_avg", "math_score"]:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)'''

'''scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)'''

#print(f"Train: {X_train.shape}, Test: {X_test.shape}")