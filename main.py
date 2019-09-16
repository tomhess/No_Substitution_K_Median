import random
from pprint import pprint
import pandas as pd
import copy
import numpy as np
import logging
from datetime import datetime
import os.path
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
import math


def skm1part1(data, m, k, delta):
    clf = cluster.KMeans(init='k-means++', n_clusters=3, random_state=42)
    clf.fit(data)
    l_centers = clf.cluster_centers_
    risk = int(clf.inertia_) / 75
    print('this is the risk:%s' % risk)
    q = calc_q(m, delta)
    # calculate the distance to the quantile points around each center
    l_distances = calc_quantile_points_around_centers(data, l_centers, q, m, k)
    return l_centers, l_distances


def calc_q(m, delta):
    return 0.1
    # return (64 * math.log((4 * m) / delta)) / m


def calc_quantile_points_around_centers(data, l_centers, q, m, k):
    l_quantile_points = []
    for i in range(k):
        l_quantile_points.append(calc_quantile_point(data, l_centers[i], q, m))
    return l_quantile_points


def calc_quantile_point(data, center, q, m):
    l_distance = np.linalg.norm(data - np.array([center]), axis=1)  # unsorted distance list
    l_distance = np.sort(l_distance)  # sort in ascending order
    q_rounding = math.ceil(m * q)  # calculate the exact location of the distance to the quantile point
    return l_distance[q_rounding]


def skm1part2(data, l_centers, l_distances, k):
    output = np.zeros((k, len(l_centers[0])))
    l_found_point = [0 for _ in range(k)]
    print(l_found_point)
    t_size = 0
    for x in data:
        for i, (center, distance) in enumerate(zip(l_centers, l_distances)):
            if (l_found_point[i] == 0) and (np.linalg.norm(x - center) <= distance):
                output[i] = x
                l_found_point[i] = 1
                t_size += 1
                print("found center number %s, this is the center: %s" % (t_size, output[i]))
                if t_size == k:
                    return output

    return output


def skm1(data, m, k, delta):
    phase_size = round(m / 2)
    # running an offline algorithm on the first half of the stream in order to compute the centers
    l_centers, l_distances = skm1part1(data[:phase_size], m, k, delta)
    print("This is the centers: %s %s %s This is the distances: %s" % ('\n', l_centers, '\n', l_distances))
    # Second part: The selection part.
    output = skm1part2(data[phase_size:], l_centers, l_distances, k)
    return output, l_centers


def main():
    iris = datasets.load_iris()
    x = iris.data  # we only take the first two features.
    y = iris.target
    print(y)
    x = pd.DataFrame(x).apply(lambda c: c / c.max(), axis=0).values  # modify the data so the data would be in [0,1]
    np.random.shuffle(x)  # random order
    x_lists = x.tolist()
    # print(round(150/2))
    # output, l_centers = skm1(x, x.shape[0], 3, 0.1)
    # print(output)
    # labels = []
    # for features in x:
    #     distances = [np.linalg.norm(features - centroid) for centroid in output]
    #     labels.append(np.argmin(distances))
    # print(len(labels))
    # inertia = np.sum((x - output[labels]) ** 2, dtype=np.float64) / 150
    # print(inertia)
    # label = []
    # for features in x:
    #     distances = [np.linalg.norm(features - centroid) for centroid in l_centers]
    #     label.append(np.argmin(distances))
    # inertia2 = np.sum((x - l_centers[label]) ** 2, dtype=np.float64) / 150
    # print(inertia2)
    # Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(x_lists, [1, 70, 140])
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    medoids = kmedoids_instance.get_medoids()
    print(' this is the mediods : %s %s' % ('\n', medoids))
    clusters = kmedoids_instance.get_clusters()
    # Show allocated clusters.
    print(clusters)
    # Display clusters.
    visualizer = cluster_visualizer_multidim()
    visualizer.append_clusters(clusters, x_lists)
    visualizer.show()
    return


if __name__ == '__main__':
    main()
