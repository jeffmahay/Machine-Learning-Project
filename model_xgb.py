import pandas as pd
import numpy as np
from preprocess import preprocess

from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# %%
# Load data and preprocess
movies = pd.read_csv('movies.csv')
movies_clean = preprocess(movies)
movies_cleaner = movies_clean.drop(columns=['name'], axis=1)
# %%
# Seperate target from the rest of the dataset
X = movies_cleaner.drop(['score', 'votes'], axis=1)
y = movies_cleaner['score']
# %%
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBRegressor(
    objective='reg:squarederror', 
    eta=0.005, 
    max_depth=10, 
    n_estimators=1500, 
    min_child_weight=1, 
    colsample_bytree=0.8, 
    colsample_bylevel=0.8, 
    colsample_bynode=0.8, 
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
    )

base_models = [
    ('xgb1', model),
    ('xgb2', model),
    ('rf', RandomForestRegressor())
]

meta_model = RidgeCV()

model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
model.fit(X_train, y_train)
# %%
def predict(model, user_input):
    predictions = np.round(model.predict(user_input), 1)
    return predictions