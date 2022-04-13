# %%
import pandas as pd 
data = {"Name": ["John", "Anna", "Peter", "Linda"],
        "Age": [24, 13, None, 33],
        "Gender": ["m", "f", "m", "f"],
        "Job": ["Programmer", "Teacher", "Programmer", "Teacher"]}

df = pd.DataFrame(data)
print(df)
# %%
#Preprocessing Pipeline: 
# Drop Name feature
# Impute ages 
# Turn Gender into binary / numeric 
# One hot encode jobs 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#Drop name feature
df = df.drop(["Name"], axis=1)

#Input the ages 
imputer = SimpleImputer(strategy="mean")
df['Age'] = imputer.fit_transform(df[['Age']])


#Turn Gender into binary / numeric
gender_dct = {"m": 0, "f": 1}
df['Gender'] = [gender_dct[g] for g in df['Gender']]

print(df)
# %%
# OneHot Encoder Jobs
encoder = OneHotEncoder()
matrix = encoder.fit_transform(df[['Job']]).toarray()
column_names = ['Programmer', 'Teacher']

for i in range(len(matrix.T)):
    df[column_names[i]] = matrix.T[i]

df = df.drop(['Job'], axis=1)
print(df)

# %%
# Putting everything in classes 
from sklearn.base import BaseEstimator, TransformerMixin

class NameDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 

    #X é o dataframe
    def transform(self, X):
        return X.drop(["Name"], axis=1)

class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 

    #X é o dataframe
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 

    #X é o dataframe
    def transform(self, X):
        # OneHot Encoder Jobs
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Job']]).toarray()
        column_names = ['Programmer', 'Teacher']

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        X = X.drop(['Job'], axis=1)
        print(X)


# %%
data = {"Name": ["John", "Anna", "Peter", "Linda"],
        "Age": [24, 13, None, 33],
        "Gender": ["m", "f", "m", "f"],
        "Job": ["Programmer", "Teacher", "Programmer", "Teacher"]}

dropper = NameDropper() 
imp = AgeImputer()
enc = FeatureEncoder() 
df2 = pd.DataFrame(data)

enc.fit_transform(imp.fit_transform(dropper.fit_transform(df2)))

#Pipelines
# %% 
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [
        ("dropper", NameDropper()),
        ("imputer", AgeImputer()),
        ("encoder", FeatureEncoder())
    ]
)
pipe.fit_transform(df2)
# %%
