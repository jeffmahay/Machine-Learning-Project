# %%
import datetime
# %%
# Function that cleans the data
def preprocess(df):
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
    df['day'] = df['date'].str.extract(r'(\d{1,2})').astype(int)
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

    def calculate_popularity(df, individual_column):
       
       weight_profit=0.1
       weight_num_movies=0.5
       threshold_value=4000000

       df['profit'] = df['gross'] - df['budget']

       individual_stats = df.groupby(individual_column).agg({
              'profit': 'mean'
       }).reset_index()

       df['weighted_mean_profit'] = df[individual_column].map(
              individual_stats.set_index(individual_column)['profit'] * weight_profit)

       df['weighted_popularity_score'] = df['weighted_mean_profit'] + weight_num_movies

       df['popular_' + individual_column] = (df['weighted_popularity_score'] >= threshold_value).astype(int)

       df.drop(['weighted_mean_profit', 'weighted_popularity_score'],axis=1, inplace=True)

       return df

    calculate_popularity(df, 'director')
    calculate_popularity(df, 'writer')
    calculate_popularity(df, 'star')
    calculate_popularity(df, 'company')

    df['season'] = df['month'].apply(lambda x: 
        '4' if x in [12, 1, 2] 
        else '1' if x in [3, 4, 5] 
        else '2' if x in [6, 7, 8] 
        else '3' if x in [9, 10, 11]
        else 'NaN'
        ).astype(int)
    
    df.drop(['director', 'writer', 'star', 'company'], axis=1, inplace=True)

    df['runtime'] = df['runtime'].astype(float)

    

    df['adult_film'] = df['rating'].isin([0, 5, 6, 7, 8, 9]).astype(int)

    df['foreign_release'] = (df['country'] != df['released_country']).astype(int)

    df['is_popular'] = ((df['popular_director'] == 1) &
                   (df['popular_writer'] == 1) &
                   (df['popular_company'] == 1)).astype(int)
        
    df['profitable'] = ((df['gross'] - df['budget']) > 0).astype(int)

    df['budget_to_gross_ratio'] = df['budget'] / df['gross']
    df['age_at_release'] = datetime.date.today().year - df['year']

    return df