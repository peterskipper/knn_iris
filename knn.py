"""Module implements a kNN Classifier for Iris dataset"""
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
from collections import Counter
import operator

IRIS = datasets.load_iris()


def plot_iris(iris=IRIS):
    """Plots the Sepal length & Width data"""
    feats = iris.data[:, :2]
    target = iris.target
    x_min, x_max = feats[:, 0].min() - .5, feats[:, 0].max() + .5
    y_min, y_max = feats[:, 1].min() - .5, feats[:, 1].max() + .5
    plt.figure(1)
    plt.scatter(feats[:, 0], feats[:, 1], c=target, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('iris_scatter.png')


def normalize_data(iris=IRIS):
    """Returns Z scores for first 2 features of data"""
    dframe = pd.DataFrame(iris.data[:, :2], columns=['slength', 'swidth'])
    norm_x = ((dframe['slength'] - dframe['slength'].mean()) /
              dframe['slength'].std())
    norm_y = ((dframe['swidth'] - dframe['swidth'].mean()) /
              dframe['swidth'].std())
    return np.asarray([norm_x, norm_y]).T


def find_dists(point, p_array):
    """
    Calculates the distance between a point and all other data
    points in the set.
    """
    return [np.linalg.norm(point - pt) for pt in p_array]


def knn(point, k=11, iris=IRIS):
    """
    kNN algorithm.  Finds the closes neighbors to point, and returns
    the majority class of those neighbors
    """
    feats = normalize_data()
    target = iris.target
    dists = find_dists(point, feats)
    closest = [x for (y, x) in sorted(zip(dists, target))][:k]
    counts = Counter(closest)
    return max(counts.iteritems(), key=operator.itemgetter(1))[0]


if __name__ == '__main__':
    plot_iris()
    point = np.random.randn(1, 2)
    guessed = knn(point)
    print 'Point: {}'.format(point)
    print 'Class: {}'.format(guessed)
