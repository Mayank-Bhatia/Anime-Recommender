import numpy as np
import pandas as pd
from surprise import KNNBasic, KNNWithMeans, KNNBaseline
from surprise import Dataset
from surprise import GridSearch, print_perf
from surprise import Reader

#the data
anime = pd.read_csv('anime.csv')
anime_rating = pd.read_csv('rating.csv')

#data cleaning
anime['type'] = anime['type'].fillna('None')
anime['genre'] = anime['genre'].fillna('None')
anime['rating'] = anime['rating'].fillna('None')
anime_rating = anime_rating[anime_rating.rating > 0] # only keep ratings between 1-10
anime_rating = anime_rating[anime_rating.user_anime_count > 19] # only keep users that have seen at least 20 anime
anime_rating = anime_rating.drop('user_anime_count', axis=1) # having served its purpose, we can drop the user count column

#load dataframe into surprise
dummy_reader = Reader(line_format='user item rating', rating_scale=(1, 10))
rating_data = Dataset.load_from_df(anime_rating[['user_id', 'anime_id', 'rating']], dummy_reader)
rating_data.split(n_folds=3)

#hyperparameter tuning
sim_options1 = {'name': 'pearson_baseline', 'user_based': False}
sim_options2 = {'name': 'msd', 'user_based': False}
sim_options3 = {'name': 'cosine', 'user_based': False}

bsl_options1 = {'method': 'als', 'learning_rate': .001}
bsl_options2 = {'method': 'sgd', 'learning_rate': .001}

param_grid = {'sim_options': [sim_options1,sim_options2,sim_options3]}

param_grid_bsl = {'sim_options': [sim_options1,sim_options2,sim_options3],
                  'bsl_options': [bsl_options1,bsl_options2]}

grid_search_basic = GridSearch(KNNBasic, param_grid, measures=['RMSE', 'FCP'], verbose=0)
grid_search_means = GridSearch(KNNWithMeans, param_grid, measures=['RMSE', 'FCP'], verbose=0)
grid_search_bsl = GridSearch(KNNBaseline, param_grid_bsl, measures=['RMSE', 'FCP'], verbose=0)

grid_search_basic.evaluate(rating_data)
grid_search_means.evaluate(rating_data)
grid_search_bsl.evaluate(rating_data)

#train model with highest performance
anime_algo = KNNBaseline(sim_options=sim_options1, bsl_options=bsl_options1)
rating_trainset = rating_data.build_full_trainset()
testing_model = anime_algo.train(rating_trainset)

def get_index(x):
    # gives index for the anime
    return anime[anime['name']==x].index.tolist()[0]

def recommend_me(a):
    print('Here are 10 anime similar to', a, ':' '\n')
    index = get_index(a)
    anime_nbrs = anime_algo.get_neighbors(index, k=10)
    
    for i in anime_nbrs[:]:
            print(anime.iloc[i]['name'], 
                  '\n' 'Genre: ', anime.iloc[i]['genre'],
                  '\n' 'Episode count: ', anime.iloc[i]['episodes'],
                  '\n' 'Rating out of 10:', anime.iloc[i]['rating'], '\n')
