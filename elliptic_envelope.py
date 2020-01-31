# -*- coding: utf-8 -*-
"""
script for drawing confidence ellipses (a.k.a. gaussian elliptic envelopes)

Created on Fri Jan 31 00:22:50 2020

@author: Kedar
"""
import numpy as np
from matplotlib import pyplot as plt
plt.ioff()
import webbrowser

# define a normal distribution that roughly spans -1 to 1
mu = 0.0
sigma = 0.35

# create some ellipse-like data, using that distribution
n_points = 2500
a = 3.0     # semi-major axis
b = 1.0     # semi-minor axis
x = a*np.random.normal(mu, sigma, n_points)
y = b*np.random.normal(mu, sigma, n_points)



# plot the points
plot_name = 'points'
auto_open = True
the_fontsize = 16
plt.figure(plot_name)
# plotting
plt.plot(x, y,'k.')
plt.xlabel('$x$', fontsize=the_fontsize)
plt.ylabel('$y$', fontsize=the_fontsize)
plt.title('$n = ' + str(n_points) + '$')
plt.axis('equal')
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name + '.png'
plt.savefig(file_name, dpi=300)
print('figure saved: ' + plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
