import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean

data = pd.read_csv('data', delim_whitespace=True)
data.columns = ['id', 'dist', 'speed']

distances = data['dist'].values.tolist()
speeds = data['speed'].values.tolist()

data['normalized_dist'] = (data.dist - data.dist.mean())/data.dist.std()
data['normalized_speed'] = (data.speed - data.speed.mean())/data.speed.std()

normalized_distances = data['normalized_dist'].values.tolist()
normalized_speeds = data['normalized_speed'].values.tolist()


plt.scatter(data["dist"], data["speed"])
plt.show()














##### FUNCTIONS #####


def guess_centroid(distances, speeds):
    plt.scatter(distances, speeds)
    plt.show()
    centroid_input = input('Input Centroid Locations: ')
    centroids = eval(centroid_input)
    return centroids

def dist(distances, speeds, centroids=None):
    if centroids is None:
        centroids = guess_centroid(distances, speeds)


    clusters = {}
    for centroid in centroids:
        clusters[centroid] = []


    for i in range(len(distances)):
        distances_to_centroids = []
        for centroid in centroids:
            distance_to_centroid = ((centroid[0] - distances[i])**2) + ((centroid[1] - speeds[i])**2)
            distances_to_centroids.append(distance_to_centroid)
        closest_centroid = centroids[distances_to_centroids.index(min(distances_to_centroids))]
        clusters[closest_centroid].append(i)


    for centroid in centroids:
        x = [distances[k] for k in clusters[centroid]]
        y = [speeds[k] for k in clusters[centroid]]
        plt.scatter(x, y)
        plt.plot([centroid[0]], [centroid[1]], 'k', marker='D')

    plt.show()
    return clusters


def calculate_new_centroids(distances, speeds, clusters):
    new_centroids = []
    for centroid in clusters:
        x_mean = mean([distances[k] for k in clusters[centroid]])
        y_mean = mean([speeds[k] for k in clusters[centroid]])
        new_centroids.append((x_mean, y_mean))
    return new_centroids


def clusters(distances, speeds, centroids=None, repetitions=4):
    for n in range(repetitions):
        clusters = dist(distances, speeds, centroids)
        centroids = calculate_new_centroids(distances, speeds, clusters)
    return clusters

################################################################
#guess_centroid(normalized_distances, normalized_speeds)

clusters(normalized_distances, normalized_speeds, ((-.5, .5), (-.5, 1.5), (2, 0), (2, 4), (1, 3)))


