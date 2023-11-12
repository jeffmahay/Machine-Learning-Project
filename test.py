
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# %%
# Load the data
movies = pd.read_csv('movies.csv')
movies.info()
# %%
movies.head()
# %%
def preprocess(df, weight_profit=0.1, weight_num_movies=0.5, threshold_value=4000000):
    # Remove NaN
    df['rating'].fillna('Unrated', inplace=True)
    df.dropna(subset=['budget', 'gross', 'company'], inplace=True)
    df.loc[df['country'].isna(), 'country'] = 'United States'
    df.loc[df['runtime'].isna(), 'runtime'] = '91.0'

    # Enumerate the released date
    valid_format = df['released'].str.contains(r'\w+ \d{1,2}, \d{4} \(.+\)')
    df = df[valid_format].copy()
    df.reset_index(drop=True, inplace=True)
    df[['date', 'released_country']] = df['released'].str.split('(', expand=True)
    df['day'] = df['date'].str.extract(r'(\d{1,2})')
    df['month'] = df['date'].str.extract(r'([A-Za-z]+)')
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month'] = df['month'].str.capitalize()
    df['month'] = df['month'].map(month_mapping)
    df.drop(columns=['released', 'date'], inplace=True)
    df['released_country'] = df['released_country'].str.rstrip(')')
    df['released_country'] = df['released_country'].str.strip()

    # Enumerate release country
    country_mapping = {
        'United States': 1, 'France': 2, 'Brazil': 3, 'Sweden': 4, 'South Korea': 5, 'Japan': 6, 'Australia': 7,
        'Netherlands': 8, 'United Kingdom': 9, 'Italy': 10, 'Canada': 11, 'Finland': 12, 'South Africa': 13,
        'Portugal': 14, 'Argentina': 15, 'Mexico': 16, 'Germany': 17, 'Hong Kong': 18, 'New Zealand': 19,
        'Greece': 20, 'Denmark': 21, 'Iceland': 22, 'Singapore': 23, 'Spain': 24, 'China': 25, 'Russia': 26,
        'Norway': 27, 'Ireland': 28, 'Austria': 29, 'Israel': 30, 'India': 31, 'Taiwan': 32, 'Poland': 33,
        'Czech Republic': 34, 'Latvia': 35, 'Philippines': 36, 'Turkey': 37, 'Bahrain': 38, 'Malaysia': 39,
        'Thailand': 40, 'Croatia': 41, 'Iran': 42, 'Bulgaria': 43, 'Lebanon': 44, 'Belgium': 45,
        'United Arab Emirates': 46, 'Bahamas': 47, 'Hungary': 48
    }
    df['released_country'] = df['released_country'].map(country_mapping)

    # Enumerate rating
    rating_mapping = {'Not Rated': 0, 'Approved': 1, 'G': 2, 'PG': 3, 'PG-13': 4, 'TV-MA': 5, 'R': 6, 'NC-17': 7, 'X': 8,
                      'Unrated': 9 }
    df['rating'] = df['rating'].map(rating_mapping)

    # Enumerate genre
    genre_mapping = {'Drama': 0, 'Adventure': 1, 'Action': 2, 'Comedy': 3, 'Horror': 4, 'Biography': 5, 'Crime': 6,
                     'Fantasy': 7, 'Animation': 8, 'Family': 9, 'Western': 10, 'Sci-Fi': 11, 'Romance': 12, 'Thriller': 13,
                     'Mystery': 14}
    df['genre'] = df['genre'].map(genre_mapping)

    # Enumerate origin country
    country_mapping = {'United Kingdom': 0, 'United States': 1, 'South Africa': 2, 'West Germany': 3, 'Canada': 4,
                       'Australia': 5, 'Italy': 6, 'South Korea': 7, 'Sweden': 8, 'Spain': 9, 'Hong Kong': 10,
                       'Mexico': 11, 'Switzerland': 12, 'France': 13, 'New Zealand': 14, 'Japan': 15, 'Yugoslavia': 16,
                       'Ireland': 17, 'Germany': 18, 'Austria': 19, 'Portugal': 20, 'China': 21, 'Taiwan': 22,
                       'Republic of Macedonia': 23, 'Russia': 24, 'Federal Republic of Yugoslavia': 25, 'Iran': 26,
                       'Czech Republic': 27, 'Denmark': 28, 'Jamaica': 29, 'Brazil': 30, 'Aruba': 31, 'Argentina': 32,
                       'India': 33, 'Netherlands': 34, 'Colombia': 35, 'Norway': 36, 'Israel': 37, 'Belgium': 38,
                       'United Arab Emirates': 39, 'Indonesia': 40, 'Hungary': 41, 'Kenya': 42, 'Iceland': 43,
                       'Chile': 44, 'Finland': 45, 'Panama': 46, 'Malta': 47, 'Lebanon': 48, 'Thailand': 49}
    df['country'] = df['country'].map(country_mapping)

    # Function to determine popularity
    def calculate_popularity(df, individual_column, weight_profit, weight_num_movies, threshold_value):
       df['profit_column'] = df['gross'] - df['budget']

       individual_stats = df.groupby(individual_column).agg({
              'profit_column': 'mean'
       }).reset_index()

       df['weighted_mean_profit'] = df[individual_column].map(
              individual_stats.set_index(individual_column)['profit_column'] * weight_profit)

       df['weighted_popularity_score'] = df['weighted_mean_profit'] + weight_num_movies

       df['popular_' + individual_column] = (df['weighted_popularity_score'] >= threshold_value).astype(int)

       df.drop(['profit_column', 'weighted_mean_profit', 'weighted_popularity_score'],axis=1, inplace=True)

       return df

    # Create popularity columns
    calculate_popularity(df, 'director', weight_profit, weight_num_movies, threshold_value)
    calculate_popularity(df, 'writer', weight_profit, weight_num_movies, threshold_value)
    calculate_popularity(df, 'star', weight_profit, weight_num_movies, threshold_value)
    calculate_popularity(df, 'company', weight_profit, weight_num_movies, threshold_value)

    # Remove names
    df.drop(['director', 'writer', 'star', 'company'], axis=1, inplace=True)

    df['year'] = df['year'].astype(int)
    df['budget'] = df['budget'].astype(float)
    df['gross'] = df['gross'].astype(float)
    df['runtime'] = df['runtime'].astype(float)


    return df
# %%
movies_clean = preprocess(movies)

# %%
movies_cleaner = movies_clean.drop(columns=['name'], axis=1)
# %%
X = movies_cleaner.drop(['score', 'votes'], axis=1)
y = movies_cleaner['score']
# %%
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=42069)
# %%
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train = norm.transform(X_train)

# transform testing dataabs
X_test = norm.transform(X_test)
# %%
model = Sequential()

model.add(Dense(128, input_dim=len(X_train[0]), activation='leaky_relu'))
model.add(Dropout(.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='leaky_relu'))
model.add(Dropout(.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='leaky_relu'))
model.add(Dropout(.3))
model.add(Dense(4, activation='leaky_relu'))

model.add(Dense(1, activation='leaky_relu'))
# %%
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])
# %%
early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=80)

history = model.fit(X_train, y_train, epochs=2000, validation_split=.35, batch_size=20, callbacks=[early_stop],shuffle=False)

hist = pd.DataFrame(history.history)
# %%
hist = hist.reset_index()
# %%
def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['index'], hist['mse'], label='Train Error')
    plt.plot(hist['index'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    # plt.ylim([0,50])
# %%
plot_history()

predictions = np.round(model.predict(X_test),1)

print(predictions)
# %%
result = mean_squared_error(y_test, predictions, squared=False)
result
# %%
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")


# %%
data = [{
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

user_input_df = pd.DataFrame(data)
# %%
new_data_clean = preprocess(user_input_df)

# Drop unnecessary columns
new_data_cleaner = new_data_clean.drop(columns=['name'], axis=1)

# Scale features using the same scaler
new_data_scaled = norm.transform(new_data_cleaner)

# Make predictions
new_predictions = np.round(model.predict(new_data_scaled), 1)

print(new_predictions)
# %%
print("Welcome! Please follow the provided prompts:")
print("--------------------------------------------")
# Get user input for column values
name = input("Enter the movie name: ")
rating = input("Enter the rating: ")
genre = input("Enter the genre: ")
year = int(input("Enter the release year: "))
released = input("Enter the date released and what country it released in (in parenthesis): ")
director = input("Enter the director: ")
writer = input("Enter the writer: ")
star = input("Enter the star: ")
country = input("Enter the origin country: ")
budget = float(input("Enter the movie budget: "))
gross = float(input("Enter the movie gross: "))
company = input("Enter the company: ")
runtime = float(input("Enter the movie runtime: "))

# Create a list of dictionaries with user input
data = [{
    'name': name,
    'rating': rating,
    'genre': genre,
    'year': year,
    'released': released,
    'director': director,
    'writer': writer,
    'star': star,
    'country': country,
    'budget': budget,
    'gross': gross,
    'company': company,
    'runtime': runtime,
}]

# Create a DataFrame from the list of dictionaries
user_input_df = pd.DataFrame(data)
# %%
# Preprocess the new dataframe
new_data_clean = preprocess(user_input_df)

# Drop unnecessary columns
new_data_cleaner = new_data_clean.drop(columns=['name'], axis=1)

# Scale features using the same scaler
new_data_scaled = norm.transform(new_data_cleaner)

# Make predictions
new_predictions = np.round(model.predict(new_data_scaled), 1)

print(new_predictions)
# %%