import pickle
from sortedcontainers import SortedList
from tqdm import tqdm
import numpy as np

base_path = "./../data/"
with open(base_path + "user2movie.json", 'rb') as f:
    user2movie = pickle.load(f)
with open(base_path + "movie2user.json", 'rb') as f:
    movie2user = pickle.load(f)
with open(base_path + "usermovie2rating.json", 'rb') as f:
    usermovie2rating = pickle.load(f)
with open(base_path + "usermovie2rating_test.json", 'rb') as f:
    usermovie2rating_test = pickle.load(f)



N = len(user2movie.keys()) + 1  #Number of users
M = max(max(movie2user.keys()), max(usermovie2rating.keys())[1]) +1  # Number of movies

k = 25  # Number of k nearest users
limit = 5  # least number of common movies between users to consider
neighbours = []
averages = []
deviations = []

def neighbours_():
    for i in tqdm(range(M)):
        # list of users who saw movie i
        users_i = movie2user[i]
        users_i_set = set(users_i)

        # calculate average and deviation of ratings given by user i
        ratings_i = {user: usermovie2rating[(user, i)] for user in users_i}
        avg_i = np.mean(list(ratings_i.values()))
        dev_i = {user: ratings_i[user] - avg_i for user in users_i}
        dev_i_values = np.array(list(dev_i.values()))
        std_i = np.sqrt(np.dot(dev_i_values, dev_i_values))
        averages.append(avg_i)
        deviations.append(dev_i)
        sl = SortedList()

        # getting users who watched movie i along with other movies j
        for j in range(M):
            if j != i:
                users_j = movie2user[j]
                users_j_set = set(users_j)
                common_users = (users_i_set & users_j_set)   # common users who watched same movie i

                # Proceed if number of common users between movie i and movie j is greater than limit
                if len(common_users) > limit:
                    # calculate average and deviation of ratings given by user j
                    ratings_j = {user: usermovie2rating[(user, j)] for user in users_j}
                    avg_j = np.mean(list(ratings_j.values()))
                    dev_j = {user: ratings_j[user] - avg_j for user in users_j}
                    dev_j_values = np.array(list(dev_j.values()))
                    std_j = np.sqrt(np.dot(dev_j_values, dev_j_values))

                    # correlation coefficient
                    numerator = sum(dev_i[m] * dev_j[m] for m in common_users)
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

def predict(i, u):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][u]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have been rated by the same user
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction



train_predictions = []
train_targets = []
for (u, m), target in tqdm(usermovie2rating.items()):
  # calculate the prediction for this movie
  prediction = predict(m, u)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (u, m), target in tqdm(usermovie2rating_test.items()):
  # calculate the prediction for this movie
  prediction = predict(m, u)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)


# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))