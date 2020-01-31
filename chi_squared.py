# -*- coding: utf-8 -*-
"""
script that explores the chi-squared distribution

Created on Fri Jan 31 00:51:31 2020

@author: Kedar
"""

import numpy as np
from scipy.special import gamma
from scipy.stats import chi2
from matplotlib import pyplot as plt
from matplotlib import cm
plt.ioff()
import webbrowser

#-----------------------------------------------------------------------------#
def chi_squared(x, k):
    '''
    this function computes discrete value of the pdf corresponding to the chi-
    squared distribution with k degrees of freedom for the values, x
    '''
    # initialize an array of zeroes of the same length as x
    p = np.zeros_like(x)
    # run through the array and compute the value of the pdf
    for i in range(len(x)):
        # treat positive and negative values appropriately
        if x[i] > 0:
            # evaluate the pdf formula
            p[i] = x[i]**(k/2 - 1)*np.exp(-x[i]/2)/(2**(k/2)*gamma(k/2))
        else:
            # for negative values, the pdf is zero, the initialized value
            pass
    # return the probability values
    return p
#-----------------------------------------------------------------------------#
    
    
# create some abscissa values
x = np.linspace(0, 10, 200)

# compute the pdfs of the chi-squared distribution for a few degrees of freedom
ks = [1,2,3,4,5,7,10,15]       # degrees of freedom to investigate
pdfs = [chi_squared(x, k) for k in ks]

# do the same thing using the the scipy function
pdfs_scipy = [chi2.pdf(x, k) for k in ks]

# plot the distribution
plot_name = 'chi-squared pdfs (mine)'
auto_open = True
the_fontsize = 16
plt.figure(plot_name)
# make a list of colors
colors = cm.rainbow_r(np.linspace(0, 1, len(ks)))
# plotting
for i in range(len(ks)):
    plt.plot(x, pdfs[i], color=colors[i], label='$k='+str(ks[i])+'$')
plt.xlabel('$x$', fontsize=the_fontsize)
plt.ylabel('$p(x)$', fontsize=the_fontsize)
plt.title('$\chi^2 \! -\! distributions \quad (subroutine)$')
plt.legend(loc='best')
plt.ylim(0, 0.5)
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name + '.png'
plt.savefig(file_name, dpi=300)
print('figure saved: ' + plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)

# plot the distribution
plot_name = 'chi-squared pdfs (scipy)'
auto_open = True
the_fontsize = 16
plt.figure(plot_name)
# make a list of colors
colors = cm.rainbow_r(np.linspace(0, 1, len(ks)))
# plotting
for i in range(len(ks)):
    plt.plot(x, pdfs_scipy[i], color=colors[i], label='$k='+str(ks[i])+'$')
plt.xlabel('$x$', fontsize=the_fontsize)
plt.ylabel('$p(x)$', fontsize=the_fontsize)
plt.title('$\chi^2 \! -\! distributions \quad (\\mathtt{scipy.stats})$')
plt.legend(loc='best')
plt.ylim(0, 0.5)
# save plot and close
print('\n\t'+'saving final image...', end='')
file_name = plot_name + '.png'
plt.savefig(file_name, dpi=300)
print('figure saved: ' + plot_name)
plt.close(plot_name)
# open the saved image, if desired
if auto_open:
    webbrowser.open(file_name)
