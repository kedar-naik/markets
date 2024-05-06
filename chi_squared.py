# -*- coding: utf-8 -*-
"""
script that explores the chi-squared distribution
"""

import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import chi2
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from matplotlib import rc, cm
import webbrowser
from distutils.spawn import find_executable
if find_executable('latex'):
    rc('text', usetex=True)


plt.ioff()


# --------------------------------------------------------------------------- #
def chi_squared_pdf(x, k):
    """
    this function computes discrete values of the pdf corresponding to the chi-
    squared distribution with k degrees of freedom for the values, x
    """
    # initialize an array of zeroes of the same length as x
    pdf = np.zeros_like(x)
    # run through the array and compute the value of the pdf
    for i in range(len(x)):
        # treat positive and negative values appropriately
        if x[i] > 0:
            # evaluate the pdf formula
            pdf[i] = x[i] ** (k / 2 - 1) * np.exp(-x[i] / 2) / (
                        2 ** (k / 2) * gamma(k / 2))
        else:
            # for negative values, the pdf is zero, the initialized value
            pass
    # return the probability values
    return pdf


# --------------------------------------------------------------------------- #
def chi_squared_cdf(x, k):
    """
    this function computes discrete values of the cdf corresponding to the chi-
    squared distribution with k degrees of freedom for the values, x. this is
    the same as the regularized gamma function (i.e. the lower incomplete
    gamma function divided by the gamma function). recall: cdf(x) is the
    probability that the random variable being modeled is less than or equal to
    x
    """
    # compute the value of the regularized gamma function. this is the cdf
    cdf = gammainc(k / 2, x / 2)
    # return the cdf values
    return cdf


# --------------------------------------------------------------------------- #
def cdf_root_function(x, k, F_desired):
    """
    this is a function for feeding to fsolve to find root, x, in the cdf of the
    chi-squared distribution with k degrees of freedom for a desired
    cumulative probability, F_desired
    """
    # compute the root equation
    return chi_squared_cdf(x, k) - F_desired


# --------------------------------------------------------------------------- #
# noinspection PyTypeChecker
def main():
    """
    main routine
    """
    # create some abscissa values
    x = np.linspace(0, 10, 200)

    # compute the pdfs of the chi-squared dist. for a few degrees of freedom
    ks = [1, 2, 3, 4, 5, 7, 10, 15]  # degrees of freedom to investigate
    pdfs = [chi_squared_pdf(x, k) for k in ks]

    # do the same thing using the the scipy function
    pdfs_scipy = [chi2.pdf(x, k) for k in ks]

    # compute the cdfs of the chi-squared dist. for a few degrees of freedom
    cdfs = [chi_squared_cdf(x, k) for k in ks]

    # do the same thing using the the scipy function
    cdfs_scipy = [chi2.cdf(x, k) for k in ks]

    # find the right-hand-side constant for an F_desired-percentage confidence
    # interval for an uncorrelated 2D dataset
    k = 2  # dimensions of the ellipsoid
    confidence_prob = 0.95  # desired confidence interval
    ellipse_rhs = fsolve(cdf_root_function, np.array(0.0),
                         args=(k, confidence_prob))[0]
    # print the ellipsoid constant found
    print('\n\t' + 60 * '-')
    print('\n\tRHS constant computation: %d-dimensional ellipsoid' % k)
    print('\n\t  desired confidence:\t\t', confidence_prob)
    print('\n\t  RHS ellipsoid constant:\t', ellipse_rhs)
    print('\n\t' + 60 * '-')

    # plot the distribution
    plot_name = 'chi-squared pdfs (mine)'
    auto_open = True
    the_fontsize = 16
    plt.figure(plot_name)
    # make a list of colors
    colors = cm.rainbow_r(np.linspace(0, 1, len(ks)))
    # plotting
    for i in range(len(ks)):
        plt.plot(x, pdfs[i], color=colors[i], label='$k=' + str(ks[i]) + '$')
    plt.xlabel('$x$', fontsize=the_fontsize)
    plt.ylabel('$p(x)$', fontsize=the_fontsize)
    plt.title('$\chi^2 \! -\! distributions \quad (subroutine)$')
    plt.legend(loc='best')
    plt.ylim(0, 0.5)
    # save plot and close
    print('\n\t' + 'saving final image...', end='')
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
        plt.plot(x, pdfs_scipy[i], color=colors[i],
                 label='$k=' + str(ks[i]) + '$')
    plt.xlabel('$x$', fontsize=the_fontsize)
    plt.ylabel('$p(x)$', fontsize=the_fontsize)
    plt.title('$\chi^2 \! -\! distributions \quad (\\mathtt{scipy.stats})$')
    plt.legend(loc='best')
    plt.ylim(0, 0.5)
    # save plot and close
    print('\n\t' + 'saving final image...', end='')
    file_name = plot_name + '.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: ' + plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)

    # plot the distribution
    plot_name = 'chi-squared cdfs (mine)'
    auto_open = True
    the_fontsize = 16
    plt.figure(plot_name)
    # make a list of colors
    colors = cm.rainbow_r(np.linspace(0, 1, len(ks)))
    # plotting
    for i in range(len(ks)):
        plt.plot(x, cdfs[i], color=colors[i], label='$k=' + str(ks[i]) + '$')
        # plot the point that was found using fsolve
    plt.plot(ellipse_rhs, confidence_prob, 'k.')
    plt.text(ellipse_rhs, confidence_prob,
             '$' + str(100 * confidence_prob) + '\%$')
    plt.xlabel('$x$', fontsize=the_fontsize)
    plt.ylabel('$F(x)$', fontsize=the_fontsize)
    plt.title('$\chi^2 \; CDFs \quad (subroutine)$')
    plt.legend(loc='best')
    # save plot and close
    print('\n\t' + 'saving final image...', end='')
    file_name = plot_name + '.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: ' + plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)

    # plot the distribution
    plot_name = 'chi-squared cdfs (scipy)'
    auto_open = True
    the_fontsize = 16
    plt.figure(plot_name)
    # make a list of colors
    colors = cm.rainbow_r(np.linspace(0, 1, len(ks)))
    # plotting
    for i in range(len(ks)):
        plt.plot(x, cdfs_scipy[i], color=colors[i],
                 label='$k=' + str(ks[i]) + '$')
    # plot the point that was found using fsolve
    plt.plot(ellipse_rhs, confidence_prob, 'k.')
    plt.text(ellipse_rhs, confidence_prob,
             '$' + str(100 * confidence_prob) + '\%$')
    plt.xlabel('$x$', fontsize=the_fontsize)
    plt.ylabel('$F(x)$', fontsize=the_fontsize)
    plt.title('$\chi^2 \; CDFs \quad (\\mathtt{scipy.stats})$')
    plt.legend(loc='best')
    # save plot and close
    print('\n\t' + 'saving final image...', end='')
    file_name = plot_name + '.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: ' + plot_name)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)


# --------------------------------------------------------------------------- #
# run if called directly
if __name__ == '__main__':
    main()
