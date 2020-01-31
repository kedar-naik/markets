# -*- coding: utf-8 -*-
"""
code from https://gist.github.com/MaxBareiss

Created on Mon Jan 28 16:50:34 2019

@author: Kedar
"""
import numpy as np
from matplotlib import pyplot as plt
import webbrowser

# Euclidean distance.
def euc_dist(pt1,pt2):
    return np.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""
def frechetDist(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)

#-----------------------------------------------------------------------------#

n_points_1 = 43
n_points_2 = 14

np.random.seed(10)

x1 = sorted(np.random.rand(n_points_1))
y1 = sorted(np.random.rand(n_points_1))

x2 = sorted(np.random.rand(n_points_2))
y2 = sorted(np.random.rand(n_points_2), reverse=True)
y2 = sorted(np.random.rand(n_points_2))
#y2 = np.random.rand(n_points_2)

scaling = 500
x1 = scaling*np.array(x1)
y1 = scaling*np.array(y1)
x2 = scaling*np.array(x2)
y2 = scaling*np.array(y2)

curve_1 = np.array(list(zip(x1,y1)))
curve_2 = np.array(list(zip(x2,y2)))

# compute the frechet distance
dist = frechetDist(curve_1, curve_2)
print(dist)

# plotting
plot_name = 'curves'
auto_open = True
plt.figure(plot_name)
plt.plot(curve_1[:,0], curve_1[:,1], 'c.-', label='$curve \;1$')
plt.plot(curve_2[:,0], curve_2[:,1], 'm.-', label='$curve \;2$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$Fr\\acute{e}chet \, distance: \;'+str(dist)+'$')
plt.legend(loc='best')
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name+'.png'
plt.savefig(file_name, dpi=300)
print('figure saved: '+plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)

    
    
    
    
