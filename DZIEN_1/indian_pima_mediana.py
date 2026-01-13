import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


col_names = ["pregnant", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "label"]
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

cols_to_impute = ["glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "pregnant"]

pima_imputed = pima.copy()

# Upewnij się, że kolumny są numeryczne (żeby nie robiły się "object")
pima_imputed[cols_to_impute] = pima_imputed[cols_to_impute].apply(pd.to_numeric, errors="coerce")

for col in cols_to_impute:
    median_no_zeros = pima_imputed.loc[pima_imputed[col] != 0, col].median()
    pima_imputed.loc[pima_imputed[col] == 0, col] = np.nan
    pima_imputed[col] = pima_imputed[col].fillna(median_no_zeros)

print(pima_imputed.head())

feature_cols = ["pregnant", "insulin", "bmi", "age", "glucose", "bp", "pedigree"]
X = pima_imputed[feature_cols]
y = pima_imputed["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)

clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"dokładność dopasowania wyników: {metrics.accuracy_score(y_test, y_pred)}")
