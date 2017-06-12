import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('anime.csv')

# missing values and 'Unknown'
df['type'] = df['type'].fillna('None')
df['genre'] = df['genre'].fillna('None')
df['rating'] = df['rating'].fillna(df['rating'].median())
episode_ = df['episodes'].replace('Unknown', np.nan)
episode_ = episode_.fillna(episode_.median())

# wordcloud
genre_list = df['genre'].to_string()
pikachu = np.array(Image.open('pikachu.jpg'))
wordcloud = WordCloud(background_color="white", mask=pikachu).generate(genre_list)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# feature preprocessing
type_ = pd.get_dummies(df['type'])
genre_ = df['genre'].str.get_dummies(sep=',')
X = pd.concat([genre_, type_, episode_, df['rating'], df['members']],axis=1)
scaled = MaxAbsScaler()
X = scaled.fit_transform(X)

# knn
recommendations = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(X)
anime_indices = recommendations.kneighbors(X)[1]

def get_index(x):
    # gives index for the anime
    return df[df['name']==x].index.tolist()[0]

def recommend_me(anime):
    print('Here are 10 anime similar to', anime, ':' '\n')
    index = get_index(anime)
    
    # ignore first entry so as to not return the queried anime as similar to itself
    for i in anime_indices[index][1:]:
            print(df.iloc[i]['name'], 
                  '\n' 'Genre: ', df.iloc[i]['genre'],
                  '\n' 'Episode count: ', df.iloc[i]['episodes'],
                  '\n' 'Rating out of 10:', df.iloc[i]['rating'], '\n')
