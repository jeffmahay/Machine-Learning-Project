# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from preprocess import preprocess

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# %%
# Load the data
movies = pd.read_csv('movies.csv')
# %%
# Clean the data
movies_clean = preprocess(movies)
# %%
movies_cleaner = movies_clean.drop(columns=['name'], axis=1)
# %%
# Seperate target from the rest of the dataset
X = movies_cleaner.drop(['score', 'votes'], axis=1)
y = movies_cleaner['score']
# %%
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)
# transform training data
X_train = norm.transform(X_train)
# transform testing dataabs
X_test = norm.transform(X_test)
# %%
# Create NN
model = Sequential()
model.add(Dense(256, input_dim=len(X_train[0]), activation='relu', kernel_regularizer=l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

# %%
# Compile & adjust learning rate
opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])
# %%
# Fit the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=50)

history = model.fit(X_train, y_train, epochs=2000, validation_split=.35, batch_size=5, callbacks=[early_stop],shuffle=False)

hist = pd.DataFrame(history.history)
# %%
# Plot epoch history
hist = hist.reset_index()
def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['index'], hist['mse'], label='Train Error')
    plt.plot(hist['index'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    # plt.ylim([0,50])

plot_history()
# %%
# Predict
predictions = np.round(model.predict(X_test),1)

predictions
# %%
# MSE & R^2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
# %%
# TEST THE MODEL
# %%
data_g = [{
    'name': "Star Wars: Episode V - The Empire Strikes Back",
    'rating': "PG",
    'genre': "Action",
    'year': 1980,
    'released': "June 20, 1980 (United States)",
    'director': "Irvin Kershner",
    'writer': "Leigh Brackett",
    'star': "Mark Hamill",
    'country': "United States",
    'budget': 18000000.0,
    'gross': 538375067.0,
    'company': "Lucasfilm",
    'runtime': 124.0,
}]

user_input_df_g = pd.DataFrame(data_g)
# %%
data_b = [{    
    'name': "Superbabies: Baby Geniuses 2",
    'rating': "PG",
    'genre': "Comedy",
    'year': 2004,
    'released': "August 27, 2004 (United States)",
    'director': "	Bob Clark",
    'writer': "Robert Grasmere",
    'star': "Jon Voight",
    'country': "Germany",
    'budget': 20000000.0,
    'gross': 9448644.0,
    'company': "ApolloMedia Distribution",
    'runtime': 88.0,
}]
user_input_df_b = pd.DataFrame(data_b)
# %%
# GOOD MOVIE
new_data_g_clean = preprocess(user_input_df_g)

# Drop unnecessary columns
new_data_g_cleaner = new_data_g_clean.drop(columns=['name'], axis=1)
new_data_g_scaled = norm.transform(new_data_g_cleaner)

# Make predictions
new_predictions_g = np.round(model.predict(new_data_g_scaled), 1)
# %%
# BAD MOVIE
new_data_b_clean = preprocess(user_input_df_b)

# Drop unnecessary columns
new_data_b_cleaner = new_data_b_clean.drop(columns=['name'], axis=1)
new_data_b_scaled = norm.transform(new_data_b_cleaner)

# Make predictions
new_predictions_b = np.round(model.predict(new_data_b_scaled), 1)

# %%
print(new_predictions_g)
print(new_predictions_b)
# %%