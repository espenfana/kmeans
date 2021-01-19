# Based on codecademy KMeans lesson
# Expanded to plot and make gif of clustering
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from copy import deepcopy
from time import sleep
import imageio #https://ndres.me/post/matplotlib-animated-gifs-easily/

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))
noise = np.random.normal(0, 0.05, sepal_length_width.shape)
sepal_length_width = sepal_length_width + noise
# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Distance formula
def distance(a, b, s=0):
  for ai, bi in zip(a,b):
    s += (ai - bi) ** 2
  return s ** 0.5

# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))
# Distances to each centroid
distances = np.zeros(k)

centroids_old = np.zeros(centroids.shape)
error = np.zeros(3)
for i in range(k):
  error[i] = distance(centroids[i], centroids_old[i])

colors = ['tab:blue', 'tab:orange', 'tab:green']
sns.set_style('darkgrid')


def kmeans_plot():
  f, ax = plt.subplots(figsize=(10,6))
  for i in range(k):
    points = sepal_length_width[labels == i]
    ax.scatter(points[:,0] ,points[:,1], c=colors[i], alpha=0.5)

  ax.scatter(centroids[:,0], centroids[:,1], marker = 'D', s=150, c=colors)
  ax.set(
    xlabel = 'sepal length (cm)',
    ylabel = 'sepal width (cm)',
    xlim = (4, 8),
    ylim = (1.8, 4.6),
    title = 'KMeans iris')
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8') # Params from #https://ndres.me/post/matplotlib-animated-gifs-easily/
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,)) # Looks complicated
  return image

# Repeat Steps 2 and 3 until convergence:
images = [] # 
while error.all() != 0:
    centroids_old = deepcopy(centroids) # Store independent copy

    # Assign to the closest centroid
    for i in range(len(samples)):
        for j in range(k):
            distances[j] = distance(sepal_length_width[i], centroids[j])
        labels[i] = np.argmin(distances)
    
    images.append(kmeans_plot())
    sleep(1)
    
    # Calculate new centroids
    for i in range(k):
        points = sepal_length_width[labels == i]
        centroids[i] = points.mean(axis=0)
        error[i] = distance(centroids[i], centroids_old[i])
        print(error)

# Create gif
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./kmeans_iris.gif', images, fps=1)


actual_labels = iris.target
#print(actual_labels)

f, ax = plt.subplots(figsize=(10,6))
for i in range(k):
  points = sepal_length_width[actual_labels == i]
  s = ax.scatter(points[:,0] ,points[:,1], c=colors[i], alpha=0.5)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Target')
plt.show()

# Plot a crosstab of species and model results
species_ = ['setosa', 'versicolor', 'virginica']
species = [species_[i] for i in iris.target]
    
df = pd.DataFrame({'labels': labels, 'species': species})
#print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)

#gr = df.groupby(['labels', 'species']).labels.count()
# pv = gr.pivot(
#   columns = 'species',
#   index = 'labels'
# )