# Pre-processing Movie Ratings Data

import pandas as pd
import numpy as np
from collections import Counter


class PreProcess:   

    def __init__(self, users, movies):
        # Subset count for #movies (m) and #users (n)
        self.n = users
        self.m = movies

        # Read CSV
        self.df = pd.read_csv("../data/rating.csv")
        self.df.drop('timestamp', axis=1, inplace=True)

        # Numbering userId starting from 0 to N-1
        self.df.userId = self.df.apply(lambda x: self.df.userId - 1)

    def subset(self, df):
        # N = df.userId.max() + 1
        # M = df.movieId.max() + 1
        user_id_count = Counter(df.userId)
        movie_id_count = Counter(df.movieId)
        user_ids_subset = [id for id, count in user_id_count.most_common(self.n)]
        movie_ids_subset = [id for id, count in movie_id_count.most_common(self.m)]
        df_subset = df[df.userId.isin(user_ids_subset) & df.movieId.isin(movie_ids_subset)].copy()
        movie_id_mapping_subset = self.mapping(df_subset.movieId)
        df_subset.movieId = df_subset.apply(lambda x: movie_id_mapping_subset[x.movieId], axis=1)
        user_id_mapping_subset = self.mapping(df_subset.userId)
        df_subset.userId = df_subset.apply(lambda x: user_id_mapping_subset[x.userId], axis=1)
        return df_subset

    @staticmethod
    def mapping(Id):
        unique_ids = np.unique(Id)
        id_mapping = {}
        count = 0
        for id in unique_ids:
            id_mapping[id] = count
            count += 1
        return id_mapping

    def main(self):
        # Mapping Movie IDs
        movie_id_mapping = self.mapping(self.df.movieId)
        self.df.movieId = self.df.apply(lambda x: movie_id_mapping[x.movieId], axis=1)

        # Creating Subset of Large dataset
        subset_df = self.subset(self.df)
        subset_df.to_csv("./small_ratings.csv", index=None)
