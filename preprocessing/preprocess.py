import pandas as pd
from sklearn.utils import shuffle
import pickle


def split_train_test(df, n):
    df = shuffle(df)
    split = int(n*(df.shape[0]))
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]
    return df_train, df_test


def user_movie_rating_dict_train(row):
    i = int(row.userId)
    j = int(row.movieId)
    if i not in user2movie.keys():
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)
    if i not in movie2user.keys():
        movie2user[i] = [j]
    else:
        movie2user[i].append(j)
    usermovie2rating[(i, j)] = row.rating


def user_movie_rating_dict_test(row):
    i = int(row.userId)
    j = int(row.movieId)
    usermovie2rating[(i, j)] = row.rating


def save_json(dict_data, filename):
    with open(filename + '.json', 'wb') as f:
    pickle.dump(dict_data, f)

df = pd.read_csv("./../data/small_rating.csv")
df_train, df_test = split_train_test(df, 0.8)
df_train.apply(user_movie_rating_dict_train, axis=1)
df_test.apply(user_movie_rating_dict_test, axis=1)

# saving to json
save_json(user2movie, "./../data/user2movie")
save_json(movie2user, "./../data/movie2user")
save_json(usermovie2rating, "./../data/usermovie2rating")
save_json(usermovie2rating_test. "./../data/usermovie2rating_test")



