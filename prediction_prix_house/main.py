import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import joblib


df=pd.read_csv('data/USA_Housing.csv')
print(df.head())
print(df.isnull().sum())
df_nv = df.drop('Address', axis=1)

X=df_nv.drop('Price', axis=1)
Y=df['Price']
print(" X et y ont été définis avec succès")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('R2 Score:', metrics.r2_score(y_test, predictions))
# Enregistrer le modèle
joblib.dump(model, 'house_model.pkl')
print("Le modèle a été enregistré avec succès !")




