# -*- coding: utf-8 -*-
"""
script for drawing confidence ellipses (a.k.a. gaussian elliptic 
envelopes)

Created on Fri Jan 31 00:22:50 2020

@author: Kedar
"""
import numpy as np
from scipy.optimize import fsolve
import os
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.patches import Ellipse
plt.ioff()
import webbrowser
import pandas as pd
from chi_squared import cdf_root_function
from sklearn.covariance import EllipticEnvelope
from distutils.spawn import find_executable
if find_executable('latex'):
    rc('text', usetex=True)
#-----------------------------------------------------------------------------#
def compute_rotation_matrix(theta):
    '''
    returns the 2x2 matrix that rotates points in the x-y plane 
    counterclockwise about the z-axis. if n points are to be rotated, they 
    shold be loaded up into an nx2 array and then muliplied on the left by the
    rotation matrix
    '''
    # assemble the rotation matrix
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta), np.cos(theta)]])
    # return the matrix
    return rotation_matrix
#-----------------------------------------------------------------------------#
def plot_points_and_ellipse(points, ellipse_info, suffix='', 
                            outlierness_process=False, outlierness=np.nan,
                            plots_directory='.'):
    '''
    given a set of points a dictionary with information about an ellipse, this
    function plots the points and the ellipse in the same figure
    '''
    if outlierness_process:
        plot_name = 'point + ellipse'
    else:
        plot_name = 'data + ellipse'
    if suffix:
        plot_name += ' - ' + suffix
    auto_open = True
    the_fontsize = 16
    plt.figure(plot_name)
    fig, ax = plt.subplots()
    # plot the points
    if outlierness_process:
        # separate and name the specific points passed in
        new_point = points[:,0]
        intersection_1 = points[:,1]
        intersection_2 = points[:,2]
        # pull out the ellipse center
        center = ellipse_info['center']
        # measure the distance between the point and the two intersections
        dist_point_ellipse_1 = np.sqrt((new_point[0]-intersection_1[0])**2 + 
                                       (new_point[1]-intersection_1[1])**2)
        dist_point_ellipse_2 = np.sqrt((new_point[0]-intersection_2[0])**2 + 
                                       (new_point[1]-intersection_2[1])**2)
        # figure out which intersection is closer to the point
        closer_intersection = np.argmin([dist_point_ellipse_1, 
                                         dist_point_ellipse_2])
        # name the intersesction that is closer to the point itself
        if closer_intersection==0:
            intersection = intersection_1
        else:
            intersection = intersection_2
        # plot the point
        plt.plot(new_point[0], new_point[1],'g*', ms=4, zorder=1, 
                 label='$\\mathrm{new\; point}$')
        # plot the center of the ellipse
        plt.plot(center[0], center[1],'r.', ms=4, zorder=1, 
                 label='$\\mathrm{center}$')
        # plot the intersection points
        plt.plot(intersection_1[0], intersection_1[1],'k.', ms=4, zorder=1, 
                     label='$\\mathrm{intersection}$')
        plt.plot(intersection_2[0], intersection_2[1],'k.', ms=4, zorder=1)
        # draw an arrow between the closer intersection and the ellipse center
        plt.annotate(s='', xy=center, xytext=intersection, 
                     arrowprops={'arrowstyle': '<->'})
        # draw and arrow between the point and the closer intersection
        plt.annotate(s='', xy=new_point, xytext=intersection, 
                     arrowprops={'arrowstyle': '<->'})
    else:
        # if we're just plotting the data and the learned frontier, pull out 
        # the x and y points
        x = points[0,:]
        y = points[1,:]
        # plot the points
        plt.plot(x, y,'k.', ms=2, zorder=1, label='$\\mathrm{data}$')
    # label the axes
    plt.xlabel('$x$', fontsize=the_fontsize)
    plt.ylabel('$y$', fontsize=the_fontsize)
    # if plotting the outlier process, then draw the ellipse behind the points
    if outlierness_process:
        ellipse_order = 0
    else:
        ellipse_order = 2
    # plot the ellipse
    ellipse_patch = Ellipse(ellipse_info['center'], 
                            width=2*ellipse_info['semi-major axis'],
                            height=2*ellipse_info['semi-minor axis'],
                            angle=180*ellipse_info['angle']/np.pi, fill=False, 
                            edgecolor='magenta', label='$' + \
                            str(100*round(ellipse_info['confidence'], 2)) + \
                            '\%\,\mathrm{confidence}$',  lw=1, 
                            zorder=ellipse_order)
    ax.add_patch(ellipse_patch)
    # write the appropriate title, depending on the content
    if outlierness_process:
        plt.title('$point\! :\, (' + str(new_point[0])+','+str(new_point[1]) +\
                  ')\qquad ' + 'outlierness ='+str(round(outlierness,3)) + '$')
    else:
        plt.title('$n =' + str(len(x)) + ',\qquad\\theta =' + \
                  str(round(ellipse_info['angle']*180/np.pi,2))+'^\\circ\! ,'+\
                  '\;2a=' + str(round(2*ellipse_info['semi-major axis'],2)) + \
                  ',\; 2b=' + str(round(2*ellipse_info['semi-minor axis'],2))+\
                  '$')
    plt.axis('equal')
    plt.legend(loc='best')
    # save plot and close
    print('\n\t'+'saving final image...', end='')
    file_name = str(Path(plots_directory).joinpath(plot_name+'.png'))
    #file_name = plots_directory + '' + plot_name + '.png'
    plt.savefig(file_name, dpi=300)
    print('figure saved: ' + plot_name)
    plt.close('all')
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(file_name)
#-----------------------------------------------------------------------------#
def fit_ellipse(x, y, confidence_interval=0.95, make_plot=True, verbose=True):
    '''
    given lists of abscissas and ordinate values (x and y) and a confidence
    interval (float between 0 and 1), this function fits a confidence ellipse
    (a.k.a. gaussian elliptic envelope) to the data. the ellipse will represent
    the given confidence value. remember, the underlying assumption of such
    ellipses is that the data are normally distributed in both dimensions. this
    is why the key mathematical insight here is that the formula of an ellipse
    is identical to the chi-squared distribution, if x and y are both assumed
    to be gaussian random variables
    - input:
      - x:                      list of abscissa values
      - y:                      list of ordinate values
      - confidence_interval:    the probability that the drawn ellipse will 
                                encapsulate a point from the dataset    
      - make_plot:              whether or not the plot the found ellipse
      - verbose:                whether or not to the print the parameters of 
                                the ellipse to the screen
    - output:
      - ellipse_dict:           dictionary holding parameters of the ellipse
                                found, i.e. center, axes, angle
    '''
    # put the x and y values into an 2-by-n array
    points = np.stack((x, y))
    # compute the mean values in both the x and y dimensions
    x_mean, y_mean = np.mean(points, axis=1)
    # compute the covariance matrix between the x and y values
    covariance_matrix = np.cov(points)
    # now, in order to figure out the orientation of the ellipse, compute the 
    # angle corresponding to the principal direction. start by computing the 
    # eigenvalues and eigenvalues of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    # find the index of corresponding to the largest eigenvalue
    i_max = np.argmax(eig_vals)
    principal_dir = eig_vecs[:,i_max]
    angle = np.arctan(principal_dir[1]/principal_dir[0])
    angle_deg = 180*angle/np.pi
    # find the rhs constant for an ellipse corresponding to the desired 
    # confidence interval. do this by solving for where the cdf of the 
    # chi-squared distribution reaches the desired confidence level. the number 
    # of degrees of freedom in the chi-squared distribution is equal to the 
    # dimensions in the data, i.e. 2. n.b. instead of doing the root-finding 
    # operation with the cdf, the rhs can also be found by simplty evaluating 
    # the following expression: ellipse_rhs = -2*np.log(1-confidence_prob)  
    k = 2
    ellipse_rhs = fsolve(cdf_root_function, 0, args=(k,confidence_interval))[0]
    # compute the lengths of the ellipse's axes
    axis_lengths = 2.0*np.sqrt(ellipse_rhs*eig_vals)
    # compute the lengths of the ellipse axes
    major_axis = max(axis_lengths)
    minor_axis = min(axis_lengths)
    # create a dictionary holding all the information required for recreating
    # the ellipse found, namely: the semi-major axis, the semi-minor axis, the
    # coordinates of the center point, the rotation angle (in radians) from the
    # positive x axis, and the confidence interval represented
    ellipse_info = {'center':           (x_mean, y_mean),
                    'semi-major axis':  major_axis/2,
                    'semi-minor axis':  minor_axis/2,
                    'angle':            angle,
                    'confidence':       confidence_interval}
    # print out the lengths of the ellipse axes and the orientation
    if verbose:
        print('\n\tellipse info:\n')
        print('\t  - center: (%.3f, %.3f)' % (x_mean, y_mean))
        print('\t  - major axis: %.3f' % major_axis)
        print('\t  - minor axis: %.3f' % minor_axis)
        print('\t  - angle: %.3f degrees' % angle_deg)
    # plot the points
    if make_plot:
        plot_points_and_ellipse(points, ellipse_info)
    # return the dictionary with the ellipse parameters
    return ellipse_info
#-----------------------------------------------------------------------------#
def use_ellipse(points, ellipse_info, visualize_process=False, verbose=False,
                plots_directory='.'):
    '''
    given the coordinate of a new point in the x-y plane and information about
    the gaussian ellipse that was trained, this function determines whether the
    point lies inside the ellipse or outside. it also returns a measure of 
    "outlierness," which proivdes a measure of how far away from the ellipse
    boundary the point actually lies
    - input:
      - points:             coordinates of the new point(s) (list, tuple, list
                            of tuples, list of lists, or two-columned array)
      - ellipse_info:       dictionary returned by fit_ellipse function
      - visualize_process:  whether or not to make a plot of the process used
                            to compute the "outlierness" metric
      - verbose:            whether to print out the results
      - plots_directory:    a directory to hold plots (if there are made)
    - output:
      - results:            a pandas dataframe, where each row contains 
                            information about a given point, namely: the x- and
                            y- coordinates, a boolean specifying whether the
                            point is an inlier, and a measure of the point's
                            "outlierness"
    '''
    # if plots are to be made, make sure there is a directory for them
    if visualize_process:
        # create the full path to the plots directory
        plots_directory = Path(os.getcwd()).joinpath(plots_directory)
        plots_directory_str = str(plots_directory)
        # if the directory doesn't exsist, make one
        if not os.path.exists(plots_directory_str):
            os.makedirs(plots_directory_str)
        # delete the contents of the directory
        for old_file in os.listdir(plots_directory_str):
            os.remove(str(plots_directory.joinpath(old_file)))
    # pull out information about the ellipse
    center = ellipse_info['center']
    a = ellipse_info['semi-major axis']
    b = ellipse_info['semi-minor axis']
    theta = ellipse_info['angle']
    # convert the point(s) to a _x2 array 
    points = np.array(points, dtype='float').reshape(-1,2)
    # make a copy of the points passed in (prior to translation and rotation)
    original_points = np.copy(points)
    # subract off the coordinates of the ellipse's center
    points -= center
    # rotate the point in the opposite direction of the ellipse's orientation
    rotation_matrix = compute_rotation_matrix(-theta)
    points = np.dot(points, rotation_matrix)
    # see if the point falls inside or outside the ellipse
    ellipse_expressions = (points[:,0]/a)**2 + (points[:,1]/b)**2
    inlier_determinations = np.array([False]*len(ellipse_expressions))
    for i in range(len(points)):
        if ellipse_expressions[i] <= 1:
            inlier_determinations[i] = True
    # coumpute the measure of "outlierness" for each point. that is: determine,
    # in a nondimensionalized way, how far outside the ellipse the point 
    # actually lies. what does that actually  mean? here, it means 
    #   (1) finding the line that passes through the point itself and the 
    #       center of the transformed ellipse (i.e. the origin)
    #   (2) find the two points on the ellipse boundary that are intersected by 
    #       this line
    #   (3) compute the "diameter" as defined by these two points
    #   (4) divide by 2 to get the "radius" 
    #   (5) compute the distance, along this line, between the point and closer 
    #       of the two intersection points
    #   (6) finally, divide the distance between the point and the ellipse by 
    #       the "radius" 
    # here's how to interpret this metric: if the point is an outlier, the 
    # metric will be positive; if it's an inlier, the metric will be negative.
    # for outliers, the larger the value, the farther the point is from the 
    # ellipse boundary. for inliers, the smallest this value can be is -1.0, 
    # which corresponds to the point coinciding with the center of the ellipse
    outliernesses = []
    for i in range(len(points)):
        point = points[i]
        slope = point[1]/point[0]
        x_intersection_1 = a/np.sqrt(1+(a*slope/b)**2)
        y_intersection_1 = slope*x_intersection_1
        x_intersection_2 = -a/np.sqrt(1+(a*slope/b)**2)
        y_intersection_2 = slope*x_intersection_2
        diameter = np.sqrt((x_intersection_2-x_intersection_1)**2 + 
                           (y_intersection_2-y_intersection_1)**2)
        radius = diameter/2
        dist_point_ellipse_1 = np.sqrt((point[0]-x_intersection_1)**2 + 
                                       (point[1]-y_intersection_1)**2)
        dist_point_ellipse_2 = np.sqrt((point[0]-x_intersection_2)**2 + 
                                       (point[1]-y_intersection_2)**2)
        dist_point_ellipse = min(dist_point_ellipse_1, dist_point_ellipse_2)
        if inlier_determinations[i]:
            dist_point_ellipse *= -1
        outlierness = dist_point_ellipse/radius
        # store the value of the outlierness
        outliernesses.append(outlierness)
        # if desired, plot the process involved with computing the outlierness
        if visualize_process:
            # load up all the relevant point into an array
            to_plot = np.array([[point[0], point[1]],
                                [x_intersection_1, y_intersection_1],
                                [x_intersection_2, y_intersection_2]])
            # apply the appropriate rotation and translation (reverse of above)
            to_plot = np.dot(to_plot, compute_rotation_matrix(theta)) + center
            # plot the points
            plot_points_and_ellipse(to_plot.T, ellipse_info, 
                                    suffix='outlierness process - (' + \
                                    str(np.round(to_plot[0][0], 2)) + ', ' + \
                                    str(np.round(to_plot[0][1], 2)) + ')',
                                    outlierness_process=True, 
                                    outlierness=outlierness,
                                    plots_directory=plots_directory)
    # create a dataframe with the results
    results = pd.DataFrame({'x':            original_points[:,0],
                            'y':            original_points[:,1],
                            'inlier':       inlier_determinations,
                            'outlierness':  outliernesses},
                            columns=['x', 'y', 'inlier', 'outlierness'])
    # if desired, print out the results
    if verbose:
        print('\nresults:')
        print(results)    
    # return the results dictionary
    return results
#-----------------------------------------------------------------------------#
def main():
    '''
    to be run if script is called directly
    '''
    # define a normal distribution that roughly spans -1 to 1
    mu = 0.0
    sigma = 0.35
    
    # create some ellipse-like data, using that distribution
    n_points = 2500
    a = 3.0     # semi-major axis
    b = 1.0     # semi-minor axis
    x = a*np.random.normal(mu, sigma, n_points)
    y = b*np.random.normal(mu, sigma, n_points)
    
    # load up the x and y points into an n-by-2 array
    points = np.vstack((x, y)).T
    
    # apply a constant-angle rotation to the data
    theta_deg = -13
    theta = np.pi*theta_deg/180.0
    
    rotation_matrix = compute_rotation_matrix(theta)
    points = np.dot(points, rotation_matrix)
    
    # apply a shift to the data point in the x and y directions
    x_shift = 5
    y_shift = -5
    points += [x_shift, y_shift]
    
    # pull out the x and y values again as lists, for demonstration purposes
    x = list(points[:,0])
    y = list(points[:,1])
    
    # fit a confidence ellipse to the data
    print('\n  - fitting a confidence ellipse to the data...')
    confidence = 0.95
    ellipse_info = fit_ellipse(x, y, confidence_interval=confidence)
    
    # [user input] create a new point to test
    print('\n  - running a new point through the ellipse...')
    new_points = [(-5,-5), (5,5), (5,-5)]
    
    # quantiatively see if the point falls within the ellipse or not
    results = use_ellipse(new_points, ellipse_info, visualize_process=True, 
                          verbose=True, plots_directory='outlierness_plots')
                          
    # print a summary note about the results
    inlier_counts = results['inlier'].value_counts()
    
    print('\n\t  - of the', len(results), 'points passed in, there are',
          inlier_counts[True], 'inliers and', inlier_counts[False], 'outliers')
    
    # fit a scikit-learn gaussian elliptic envelope to the data
    print('\n  - fitting a scikit-learn gaussian ellipse to the data...')
    detector = EllipticEnvelope(contamination=1-confidence)
    detector.fit(points)
    
    # run the new point through the detector
    print('\n  - running a new point through the scikit-learn ellipse...')    
    new_points = np.array(new_points).reshape(-1,2)
    inlier_sk = detector.predict(new_points)
    mahalanobis_score = detector.score_samples(new_points)
    print('\n\t  - inlier:', inlier_sk)
    print('\t  - mahalanobis score:', mahalanobis_score)
        
    print('\n\tN.B. Although the Mahalanobis distances (a.k.a. "scores") ' + \
          '\n\tcomputed by scikit-learn do provide a statistically ' + \
          '\n\tmeaningful metric of how far away from the center of the ' + \
          '\n\tellipse a point lies, it doesn\'t provide any information ' + \
          '\n\tabout whether the point is an inlier or an outlier! So, it ' + \
          '\n\tmakes more sense to just use my implementation and the ' + \
          '\n\t"outlierness" metric, which spans [-1, inf): postive values ' +\
          '\n\timply outliers, negative values imply inliers, and a value ' + \
          '\n\tof -1 corresponds to the center of the ellipse.\n')
#-----------------------------------------------------------------------------#
# top-level invoking code
if __name__ == '__main__':
    main()
