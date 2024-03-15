# Tratamiento de datos
import pandas as pd
import numpy as np
from joblib import dump
import pickle
# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
# Estadistica
from scipy.stats import norm, shapiro
# Modelos
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

# Importar datos
df1 = pd.read_csv('src/data/raw/used_cars.csv').drop(columns='Unnamed: 0')
df1.columns = df1.columns.str.lower()

# Tratamiendo de datos

# Crear columna marca y modelo
df1['brand'] = df1['name'].str.split().str[0]
df1['model'] = df1['name'].str.split().str[1]
# Eliminiar columna 'new_price', falta el 86% de los datos, no es razonable imputarlo.
df1.drop(columns='new_price', inplace=True)
# Quitar str de las columnas mileage, engine y power y pasar a float.
df1['mileage'] = df1['mileage'].str.replace('kmpl', ' ').str.replace('km/kg', ' ')
df1['engine'] = df1['engine'].str.replace('CC', ' ')
df1['power'] = df1['power'].str.replace('bhp', ' ')
df1['mileage'] = pd.to_numeric(df1['mileage'])
df1['engine'] = pd.to_numeric(df1['engine'])
df1['power'] = pd.to_numeric(df1['power'])
# Mapeo columnas fuel_type, transmission y owner_type pasa pasar a numerico.
df1['fuel_type'] = pd.to_numeric(df1['fuel_type'].map({'Diesel' : 0, 'Petrol': 1, 'Electric': 2}))
df1['transmission'] = pd.to_numeric(df1['transmission'].map({'Automatic' : 0, 'Manual': 1}))
df1['owner_type'] = pd.to_numeric(df1['owner_type'].map({'Fourth & Above' : 0, 'Third': 1, 'Second': 2, 'First': 3}))
# Pasar la columna price a euros.
cambio = 89.15
df1['price'] = (df1['price'] * 100000) / cambio
# Imputar mediana en los Nans de las columnas engine, power y seats e imputar media en los Nans de mileage
engine_por_model = df1.groupby('model')['engine'].median()
df1['engine'] = df1.apply(lambda row: engine_por_model[row['model']] if pd.isna(row['engine']) else row['engine'], axis=1)
power_por_model = df1.groupby('model')['power'].median()
df1['power'] = df1.apply(lambda row: power_por_model[row['model']] if pd.isna(row['power']) else row['power'], axis=1)
seats_por_model = df1.groupby('model')['seats'].median()
df1['seats'] = df1.apply(lambda row: seats_por_model[row['model']] if pd.isna(row['seats']) else row['seats'], axis=1)
mileage_por_model = df1.groupby('model')['mileage'].mean()
df1['mileage'] = df1.apply(lambda row: seats_por_model[row['model']] if pd.isna(row['mileage']) else row['mileage'], axis=1)
# Eliminar columnas innecesarias
df1.drop(columns=['location', 'name', 'brand', 'model'], inplace=True)
# Quitar outliers en kilometers_driven
outliers = df1[df1['kilometers_driven'] > 150000]
total_datos = len(df1)
datos_por_encima_150k = len(outliers)
porcentaje_por_encima_150k = (datos_por_encima_150k / total_datos) * 100
df1 = df1[df1['kilometers_driven'] <= 150000]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns=['price']), df1['price'], test_size=0.2, random_state=73)

# Escalado y dummies
esc = MinMaxScaler()
X_train_mm = esc.fit_transform(X_train)
X_test_mm = esc.transform(X_test)

# Entrenamiento del modelo
catb = CatBoostRegressor(verbose=False, save_snapshot=False)
catb.fit(X_train_mm, y_train)

# Optimización del modelo
param_grid = {
    'depth': [8],
    'learning_rate': [0.15],
    'iterations': [150],
    'l2_leaf_reg': [1],
    'bagging_temperature': [0.0],
    'random_strength': [1],
    'border_count': [128],
    'subsample': [1.0]
}
catb = CatBoostRegressor(verbose=False)

grid_search = GridSearchCV(estimator=catb, param_grid=param_grid, cv=3, scoring='r2')

grid_search.fit(X_train_mm, y_train)

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predicción
y_pred = best_model.predict(X_test_mm)

# Guardar el modelo
with open('src/model/modelo_CB_entrenado.pkl', 'wb') as archivo:
    pickle.dump(best_model, archivo)