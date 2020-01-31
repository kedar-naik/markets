# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 14:31:38 2018

@author: Kedar
"""

import numpy as np
from matplotlib import pyplot as plt
import webbrowser
from sklearn.cluster import KMeans

# line parameters
m = 10
b_spacing = 600
b_1 = -1
b_2 = b_1 + b_spacing
b_3 = b_2 + b_spacing

# create the lines
x = np.linspace(18, 65, 50)
y_1 = m*x + b_1
y_2 = m*x + b_2
y_3 = m*x + b_3
y_1_noise = y_1 + np.random.randn(len(x))*b_spacing/3
y_2_noise = y_2 + np.random.randn(len(x))*b_spacing/3
y_3_noise = y_3 + np.random.randn(len(x))*b_spacing/3
x = np.hstack((x,x))
y_1 = np.hstack((y_1,y_1_noise))
y_2 = np.hstack((y_2,y_2_noise))
y_3 = np.hstack((y_3,y_3_noise))

# make a list of all the points
x_values = np.hstack((np.hstack((x,x)),x))
y_values = np.hstack((np.hstack((y_1,y_2)),y_3))
X = np.vstack((x_values, y_values)).transpose()

# do a k-means clustering on the points
kmeans = KMeans(n_clusters=3).fit(X)

# make a list of true labels
clusters_true = len(x)*[0] + len(x)*[1] + len(x)*[2]

# determine the order of the classes assigned
cluster_numbers = [kmeans.labels_[0]]
for cluster_number in kmeans.labels_[1:]:
    if not cluster_number in cluster_numbers:
        cluster_numbers.append(cluster_number)

# recast the clusters found to be in order
clusters_found = [cluster_numbers.index(cluster) for cluster in kmeans.labels_]

# find the indices of the points that fell into the wrong cluster
incorrect_indices = []
for i in range(len(x_values)):
    if not clusters_true[i]==clusters_found[i]:
        incorrect_indices.append(i)

# record the x and y coordinates of points that were placed in the wrong group
x_values_wrong = x_values[incorrect_indices]
y_values_wrong = y_values[incorrect_indices]

# compute the accuracy
accuracy = (len(x_values)-len(x_values_wrong))/len(x_values)
print('\n\tclustering accuracy: %.1f%%' % (100*accuracy))

# plot the points
plot_name = 'noisy lines'
plt.figure()
plt.plot(x_values, y_values, 'k.')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.title('$Points$')
filename = plot_name + '.png'
plt.savefig(filename, dpi=400)
plt.close()
webbrowser.open(filename)

# plot the lines
plot_name = 'noisy lines - correct labels'
plt.figure()
plt.plot(x, y_1, 'g.', label='$y_1$')  
plt.plot(x, y_2, 'b.', label='$y_2$')  
plt.plot(x, y_3, 'm.', label='$y_3$')  
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.title('$True \; Clusters$')
plt.legend(loc='best')
filename = plot_name + '.png'
plt.savefig(filename, dpi=400)
plt.close()
webbrowser.open(filename)

# plot the clusters
plot_name = 'noisy lines - clustered'
plt.figure()
colors = ['g', 'b', 'm']
for i in range(len(x_values)):
    plt.plot(x_values[i], y_values[i], colors[kmeans.labels_[i]]+'.')
x_centers = kmeans.cluster_centers_[:,0]
y_centers = kmeans.cluster_centers_[:,1]
plt.plot(x_centers, y_centers, 'ro',label='$centroids$')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.legend(loc='best')
plt.title('$Clusters \; Found$')
filename = plot_name + '.png'
plt.savefig(filename, dpi=400)
plt.close()
webbrowser.open(filename)

# plot the points
plot_name = 'misclustered points'
plt.figure()
plt.plot(x_values, y_values, 'k.')
plt.plot(x_values_wrong, y_values_wrong, 'r.')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.title('$Misclustered \; Points \qquad Accuracy:\, %.1f\%%$' %(100*accuracy))
filename = plot_name + '.png'
plt.savefig(filename, dpi=400)
plt.close()
webbrowser.open(filename)
