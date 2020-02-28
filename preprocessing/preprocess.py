# Pre-processing Movie Ratings Data

import pandas as pd

class PreProcess:

    def __init__(self):
       None

    def subset(self):


    def mapping(self, df):
        df


df = pd.read_csv("../data/rating.csv")
df.drop('timestamp', axis=1, inplace=True)

# Numbering userId starting from 0 to N-1
df.userId = df.apply(lambda x: df.userId - 1 )

# Mapping MovieId
unique_movie_ids = np.unique(df.movieId)
movie_id_mapping = {}
count=0
for id in unique_movie_ids:
    movie_id_mapping[id] = count
    count+=1



