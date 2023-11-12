import pandas as pd
from preprocess import preprocess


def input_diologue():
    # Get user input for column values
    print("Welcome! Please follow the provided prompts:")
    print("--------------------------------------------")
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

    # Preprocess the new dataframe
    user_input_clean = preprocess(user_input_df)

    user_input_cleaner = user_input_clean.drop(columns=['name'], axis=1)

    return user_input_cleaner
    
