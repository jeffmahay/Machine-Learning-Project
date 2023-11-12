# Overview

The purpose of this program is to predict the critic score for a movie on IMDB. The data was gathered from [Kaggle](https://www.kaggle.com/datasets/danielgrijalvas/movies/data). The model used to predict is an XGBoost model used with stacked regression. Initially another model utilizing Neural Networks was used, but XGBoost ended up being better suited for this dataset. The original purpose for this project was to get more practice using neural networks, and this was accomplished even though the final product wasn't a NN, as it is important to learn when to use one and when not to. 

As for the data, it includes 6,820 movies from IMDB. Each column is data scraped from the site, including:
budget: the budget of a movie. Some movies don't have this, so it appears as 0

company: the production company

country: country of origin

director: the director

genre: main genre of the movie.

gross: revenue of the movie

name: name of the movie

rating: rating of the movie (R, PG, etc.)

released: release date and what country it released in (Month Day, Year (Country))

runtime: duration of the movie

score: IMDb user rating

votes: number of user votes

star: main actor/actress

writer: writer of the movie

year: year of release

[Software Demo Video](https://youtu.be/wbf4iNFgxKI)

# Data Analysis Results

Many answers I found by doing this analysis relate to tuning hyperparameters for machine learning algorithms. I knew that it is important to change them, but now I better understand what changing each hyperparameter means for the model. For example, lowering the number of nodes helps the model from overfitting. 

As for questions, I wonder if there is an efficient way to tell if a movie is a sequel/in a franchise based solely from the title. Being able to indicate if a movie is a sequel or not could have a large impact on the prediction abilities of the model. 
# Development Environment

{Describe the tools that you used to develop the software}
* VSCode
* Python
* Scikit Learn
* Pandas
* Seaborn
* XGBoost
# Useful Websites

{Make a list of websites that you found helpful in this project}
* [Kaggle](https://www.kaggle.com/datasets/danielgrijalvas/movies/data)
* [ChatGPT](https://chat.openai.com/)
* [BYUI CSE 450 Module 5](https://byui-cse.github.io/cse450-course/module-05/)
* [BYUI CSE 450 Module 3](https://byui-cse.github.io/cse450-course/module-03/)
* [StackOverflow](https://stackoverflow.com/)
# Future Work

* Transfer the code into a web app
* Utilize web scrapping to gather new movie information
* Implement a function to add each user input into the dataset for dynamic learning