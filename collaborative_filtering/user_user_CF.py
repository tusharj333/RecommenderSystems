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

N = max(user2movie.keys()) + 1  # Number of Users
M = max(max(movie2user.keys()), max(usermovie2rating.keys())[1]) + 1  # Number of Movies

k = 25  # Number of k nearest users
limit = 5  # least number of common movies between users to consider
neighbours = []
averages = []
deviations = []


def neighbours_():
    for i in tqdm(range(N)):
        # list of movies watched by user i
        movies_i = user2movie[i]
        movies_i_set = set(movies_i)

        # calculate average and deviation of ratings given by user i
        ratings_i = {movie: usermovie2rating[(i, movie)] for movie in movies_i}
        avg_i = np.mean(list(ratings_i.values()))
        dev_i = {movie: ratings_i[movie] - avg_i for movie in movies_i}
        dev_i_values = np.array(list(dev_i.values()))
        std_i = np.sqrt(np.dot(dev_i_values, dev_i_values))
        averages.append(avg_i)
        deviations.append(dev_i)
        sl = SortedList()

        # getting movies watched by other users j
        for j in range(N):
            if j != i:
                movies_j = user2movie[j]
                movies_j_set = set(movies_j)
                common_movies = (movies_i_set & movies_j_set)

                # Proceed if number of common movies between user i and user j is greater than limit
                if len(common_movies) > limit:
                    # calculate average and deviation of ratings given by user j
                    ratings_j = {movie: usermovie2rating[(j, movie)] for movie in movies_j}
                    avg_j = np.mean(list(ratings_j.values()))
                    dev_j = {movie: ratings_j[movie] - avg_j for movie in movies_j}
                    dev_j_values = np.array(list(dev_j.values()))
                    std_j = np.sqrt(np.dot(dev_j_values, dev_j_values))

                    # correlation coefficient
                    numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
                    w_ij = numerator / (std_i * std_j)

                    # insert coefficients into sorted list (top 25 neighbours)
                    if len(sl) < k:
                        sl.add((-w_ij, j))
                    else:
                        break

        neighbours.append(sl)
    return neighbours


neighbors = neighbours_()


# using neighbors, calculate train and test MSE

def predict(i, m):
    # calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        # remember, the weight is stored as its negative
        # so the negative of the negative weight is the positive weight
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # neighbor may not have rated the same movie
            # don't want to do dictionary lookup twice
            # so just throw exception
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)  # min rating is 0.5
    return prediction


train_predictions = []
train_targets = []
for (i, m), target in tqdm(usermovie2rating.items()):
    # calculate the prediction for this movie
    prediction = predict(i, m)

    # save the prediction and target
    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
print("test data")
for (i, m), target in tqdm(usermovie2rating_test.items()):
    # calculate the prediction for this movie
    prediction = predict(i, m)

    # save the prediction and target
    test_predictions.append(prediction)
    test_targets.append(target)


# calculate accuracy
def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t) ** 2)


print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))
