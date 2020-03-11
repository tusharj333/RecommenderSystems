import pickle
from sortedcontainers import SortedList
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


base_path = "/content/drive/My Drive/My/recommender_system/"
with open(base_path + "user2movie.json", 'rb') as f:
    user2movie = pickle.load(f)
with open(base_path + "movie2user.json", 'rb') as f:
    movie2user = pickle.load(f)
with open(base_path + "usermovie2rating.json", 'rb') as f:
    usermovie2rating = pickle.load(f)
with open(base_path + "usermovie2rating_test.json", 'rb') as f:
    usermovie2rating_test = pickle.load(f)


N = max(user2movie.keys()) +1    # Number of Users
M = max(max(movie2user.keys()), max(usermovie2rating.keys())[1]) +1 # Number of Movies
print("N:", N, "M:", M)

# initialize vairables
K =10 ## latent factors
W = np.random.randn(N,K)  # User matrix
b = np.zeros(N)           # user bias
U = np.random.randn(M,K)  # Item matrix (Movies)
c = np.zeros(M)           # Item bias
mu = np.mean(list(usermovie2rating.values()))    ## Average ratings of all movies


def get_loss(d):
  N = float(len(d))
  sse = 0       #sum of squared errors
  for x, rating in d.items():
    i,j=x
    prediction = np.dot(W[i], U[j]) + b[i] + c[j] + mu
    sse+= (prediction - rating)**2
  return sse/N


# train the parameters
epochs = 25
reg =20. # regularization penalty
train_losses = []
test_losses = []
for epoch in tqdm(range(epochs)):
  print("epoch:", epoch)
  epoch_start = datetime.now()
  # perform updates

  # update W and b
  t0 = datetime.now()
  for i in range(N):
    # for W
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for b
    bi = 0
    for j in user2movie[i]:
      r = usermovie2rating[(i,j)]
      matrix += np.outer(U[j], U[j])
      vector += (r - b[i] - c[j] - mu)*U[j]
      bi += (r - W[i].dot(U[j]) - c[j] - mu)

    # set the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + reg)

    if i % (N//10) == 0:
      print("i:", i, "N:", N)
  print("updated W and b:", datetime.now() - t0)

  # update U and c
  t0 = datetime.now()
  for j in range(M):
    # for U
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for c
    cj = 0
    try:
      for i in movie2user[j]:
        r = usermovie2rating[(i,j)]
        matrix += np.outer(W[i], W[i])
        vector += (r - b[i] - c[j] - mu)*W[i]
        cj += (r - W[i].dot(U[j]) - b[i] - mu)

      # set the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie2user[j]) + reg)

      if j % (M//10) == 0:
        print("j:", j, "M:", M)
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  print("updated U and c:", datetime.now() - t0)
  print("epoch duration:", datetime.now() - epoch_start)


  # store train loss
  t0 = datetime.now()
  train_losses.append(get_loss(usermovie2rating))

  # store test loss
  test_losses.append(get_loss(usermovie2rating_test))
  print("calculate cost:", datetime.now() - t0)
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])



print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()