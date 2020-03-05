import pickle
from sortedcontainers import SortedList
from tqdm import tqdm
import numpy as np

with open("./../data/user2movie.json", 'rb') as f:
    user2movie = pickle.load(f)
with open("./../data/movie2user.json", 'rb') as f:
    movie2user = pickle.load(f)
with open("./../data/usermovie2rating.json", 'rb') as f:
    usermovie2rating = pickle.load(f)
with open("./../data/usermovie2rating_test.json", 'rb') as f:
    usermovie2rating_test = pickle.load(f)


N = len(user2movie.keys()) + 1  #Number of users
M = max(max(movie2user.keys()), max(usermovie2rating.keys())[1]) +1  # Number of movies

