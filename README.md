# Anime-Recommender
Anime-Recommender

An anime recommendation system using Kaggle's [Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database).

The Dataset

This dataset contains information on user preference data from 73,516 users on 12,294 anime, found on myanimelist.net. Each user is able to add anime to their completed list and give it a rating. This dataset is a compilation of those ratings, using the following features:

Anime.csv

anime_id - myanimelist.net's unique id identifying an anime <br>
name - full name of anime <br>
genre - comma separated list of genres for this anime <br>
type - movie, TV, OVA, etc <br>
episodes - how many episodes in this show. (1 if movie) <br>
rating - average rating out of 10 for this anime <br>
members - number of community members that are in this anime's "group" 

Rating.csv

user_id - non identifiable randomly generated user id <br> 
anime_id - the anime that this user has rated <br>
rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating) <br>

The Goal

To build a recommendation system based on user viewing history by using nearest neighbors and collaborative filtering approach. See the notebook for a detailed analysis; to view just the code, see the individual .py files.

Feedback appreciated!
