# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:29:54 2018

@author: Kedar
"""
import numpy as np
from functools import reduce
from itertools import product
from matplotlib import pyplot as plt
import webbrowser
from datetime import datetime
plt.ioff()
#-----------------------------------------------------------------------------#
def print_matrix(A, name='A', n_indents=1):
    '''
    given a numpy matrix, A, print it to the screen with nice indentation
    '''
    # convert input to a numpy array, in case it isn't already
    A = np.array(A)
    # format and print the input, depending on whether it's an array or scalar
    if len(A.shape) >= 2:
        print('\n' + n_indents*'\t' + name + ' =\n' + (n_indents + 1)*'\t', 
              str(A).replace('\n', '\n' + (n_indents + 1)*'\t' + ' '))
    else:
        print('\n' + n_indents*'\t' + name + ' =', str(A))
#-----------------------------------------------------------------------------#
def extended_exp(x):
    '''
    this subroutine makes a simple extension to the exp function: whenever a 
    nan is passed in as an argument, the function returns zero, as opposed to
    nan. the assumption here is that in initial nan in the argument is due to
    a ln(0) value. see mann, algorithm 1
    '''
    # use the standard exp function first
    exp_x = np.exp(x)
    # run through the array and figure out where there are nans or infs
    problem_locs = np.argwhere((np.isnan(exp_x)) | (abs(exp_x)==np.inf))
    # run through these locations and replace the values with zeros
    for loc_array in problem_locs:
        # if we're just working with a scalar that's a nan, then just reassign
        # the value computed by np.exp directly
        if type(exp_x) == np.float64:
            exp_x = 0.0
        else:
            # otherwise, convert the individual arrays to tuples in order to 
            # use the output of argwhere as array indices (numpy accepts tuples 
            # as indices)
            exp_x[tuple(loc_array)] = 0.0
    # return the exponentiated array
    return exp_x
#-----------------------------------------------------------------------------#
def extended_ln(x):
    '''
    this subroutine makes a simple extension to the natural-log function: 
    whenever a zero is passed in, nan is returned (instead of -inf) without a 
    runtime warning. whenever a negative number is passed in an error is raised 
    as opposed to returning nan with a runtime warning. see mann, algorithm 2
    '''
    # convert the input to a numpy array
    x = np.array(x)
    # add a dimension, if a scalar has been passed in
    if x.size == 1:
        x = np.expand_dims(x, axis=0)
    # initialize the return array with nans
    ln_x = np.nan*np.ones_like(x)
    # pull out a tuple with the dimensions of the input matrices
    dims = x.shape
    # we want to run through the array here, element-by-element. we'll do this
    # by first making a list of all the indices that need to be visited, i.e.
    # all the indices encompassed by an array with dimensions in dims. this can
    # be done in a single line: first, use the map function to create a list of 
    # range objects corresponding to each of the dimensions found; then, take 
    # this list of range objects and apply it as an argument to the itertools 
    # product function. the result is a list of tuples of the indices
    indices = list(product(*map(range, dims)))    
    # run through the arrays, element-by-element
    for index in indices:
        # pull out the value at this index
        x_val = x[index]
        # if the element is zero, then return nan
        if x_val == 0:
            ln_x[index] = np.nan
        # if the element is positive, then compute the log as usual
        elif x_val > 0:
            ln_x[index] = np.log(x_val)
        else:
            # if the input is negative, then raise an error
            if x_val < 0:
                raise ValueError('\n\n\tcannot take natural log of a ' + \
                                 'negative value!')
            else:
                #if it's a nan or an inf or something, raise another error
                raise ValueError('\n\n\tcannot take the natural log of ' + \
                                 str(x_val))
    # return the scalar or array of logarithms
    if x.size == 1:
        return float(ln_x.squeeze())
    else:
        return ln_x
#-----------------------------------------------------------------------------#
def ln_sum(ln_x, ln_y):
    '''
    given ln(x) and ln(y), this funtion returns ln(x+y) in a way that avoid 
    numerical underflow/overflow issues. if either ln_x or ln_y is nan, then 
    then the non-nan input is returned, i.e. ln(x+y) = ln(y) if ln(x) = nan.
    if both ln_x and ln_y are positive, then the log-sum-exp trick is used to
    compute ln(x+y), i.e. ln(x+y) = ln(x) + ln(1+e^(ln(y)-ln(x))), where ln(x)
    is bigger than ln(y). see mann, algorithm 3
    '''
    # convert the inputs to a numpy arrays
    ln_x = np.array(ln_x)
    ln_y = np.array(ln_y)
    # to each input, add a dimension, if a scalar has been passed in
    if ln_x.size == 1:
        ln_x = np.expand_dims(ln_x, axis=0)
    if ln_y.size == 1:
        ln_y = np.expand_dims(ln_y, axis=0)
    # make sure the two inputs are the same shape
    if not ln_x.shape == ln_y.shape:
        raise ValueError('\n\n\tcannot take a product of two arrays of ' + \
                         'different shapes: \n\tln_x.shape = ' + \
                         str(ln_x.shape) + '\n\tln_y.shape ='+str(ln_y.shape))
    # create an array to store the product, initialize it with nans
    ln_x_plus_y = np.nan*np.ones_like(ln_x)
    # pull out a tuple with the dimensions of the input matrices
    dims = ln_x.shape
    # we want to run through the arrays here, element-by-element. we'll do this
    # by first making a list of all the indices that need to be visited, i.e.
    # all the indices encompassed by an array with dimensions in dims. this can
    # be done in a single line: first, use the map function to create a list of 
    # range objects corresponding to each of the dimensions found; then, take 
    # this list of range objects and apply it as an argument to the itertools 
    # product function. the result is a list of tuples of the indices
    indices = list(product(*map(range, dims)))    
    # run through the arrays, element-by-element
    for index in indices:
        # pull out and store the values from each array at this index
        ln_x_val = ln_x[index]
        ln_y_val = ln_y[index]
        # if both inputs are nan, return nan
        if np.isnan(ln_x_val) and np.isnan(ln_y_val):
            ln_x_plus_y_val = np.nan
        # if one of the inputs is nan, then reutrn the other
        elif np.isnan(ln_x_val):
            ln_x_plus_y_val = ln_y_val
        elif np.isnan(ln_y_val):
            ln_x_plus_y_val = ln_x_val
        # in the case where both arguments aren't nans, then figure out which 
        # one is larger and then use the log-sum-exp trick
        elif ln_x_val > ln_y_val:
            ln_x_plus_y_val = ln_x_val+extended_ln(1+np.exp(ln_y_val-ln_x_val))
        else:
            ln_x_plus_y_val = ln_y_val+extended_ln(1+np.exp(ln_x_val-ln_y_val))
        # record the sum value found
        ln_x_plus_y[index] = ln_x_plus_y_val
    # return either the resulting array or scalar
    if ln_x_plus_y.size == 1:
        return float(ln_x_plus_y.squeeze())
    else:
        return ln_x_plus_y
#-----------------------------------------------------------------------------#
def ln_product(ln_x, ln_y):
    '''
    given ln(x) and ln(y), this funtion returns ln(xy). if either of the inputs
    are nan, then the returned value is nan. see mann, algorithm 4
    '''
    # convert the inputs to a numpy arrays
    ln_x = np.array(ln_x)
    ln_y = np.array(ln_y)
    # to each input, add a dimension, if a scalar has been passed in
    if ln_x.size == 1:
        ln_x = np.expand_dims(ln_x, axis=0)
    if ln_y.size == 1:
        ln_y = np.expand_dims(ln_y, axis=0)
    # make sure the two inputs are the same shape
    if not ln_x.shape == ln_y.shape:
        raise ValueError('\n\n\tcannot take a product of two arrays of ' + \
                         'different shapes: \n\tln_x.shape = ' + \
                         str(ln_x.shape) + '\n\tln_y.shape ='+str(ln_y.shape))
    # create an array to store the product, initialize it with nans
    ln_xy = np.nan*np.ones_like(ln_x)
    # pull out a tuple with the dimensions of the input matrices
    dims = ln_x.shape
    # we want to run through the arrays here, element-by-element. we'll do this
    # by first making a list of all the indices that need to be visited, i.e.
    # all the indices encompassed by an array with dimensions in dims. this can
    # be done in a single line: first, use the map function to create a list of 
    # range objects corresponding to each of the dimensions found; then, take 
    # this list of range objects and apply it as an argument to the itertools 
    # product function. the result is a list of tuples of the indices
    indices = list(product(*map(range, dims)))    
    # run through the arrays, element-by-element
    for index in indices:
        # if both inputs are valid, i.e. not nans, then compute the product 
        # using standard logarithm rules
        if not np.isnan(ln_x[index]) and not np.isnan(ln_y[index]):
            ln_xy[index] = ln_x[index] + ln_y[index]
    # return either the resulting array or scalar
    if ln_xy.size == 1:
        return float(ln_xy.squeeze())
    else:
        return ln_xy
#-----------------------------------------------------------------------------#
def forward_basic(O, A, B, pi, V, return_matrix=False):
    '''
    compute the likelihood (i.e. the probability) of the given sequence, O, 
    with respect to the hidden markov model defined by the transition
    probabilities in A and the emission probabilites in B. do so using the 
    forward algorithm. return the forward-path probabilities, if desired
    - input:
      - O:              the observation sequence under consideration
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - pi:             the vector of the initial-state probabilities
      - V:              the vocabulary of ordered possible observations
      - return_matrix:  return the matrix of forward-path probabilities (alpha)
    - output:
      - P_O:            the likelihood of the given sequence of observations
      - alpha:          N-by-T matrix of forward-path probabilities
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # create a matrix to hold the forward path probabilities
    alpha = np.zeros((N,T))
    # fill in the first column of matrix with the initial values for the 
    # forward path probabilties
    first_o_index = V.index(O[0])
    alpha[:,0] = np.multiply(pi.reshape((1,N)),B[:,first_o_index])
    # using recursion, fill in the rest of the forward-path probs
    for j in range(1,T):
        # find the index of the observation at time step j
        o_index = V.index(O[j])
        # walk down the j-th column
        for i in range(N):
            # compute alpha value by summing over all possible previous states
            # N.B. alpha[i,j] is the probability of being in the i-th state at
            # the j-th time step and seeing the first j observations
            for i_previous in range(N):
                alpha[i,j] += alpha[i_previous,j-1]*A[i_previous,i]*B[i,o_index]
    # the probability of the entire sequence is found by summing down the last 
    # column of the matrix of forward-path probabilities
    P_O = np.sum(alpha[:,-1])
    # return the probability of the observation sequence and, if desired, the
    # matrix of forward-path probabilities
    if return_matrix:
        return P_O, alpha
    else:
        return P_O
#-----------------------------------------------------------------------------#
def forward_log(O, A, B, pi, V, return_matrix=False):
    '''
    compute the log likelihood (i.e. the natural log of the probability) of the 
    given sequence, O, with respect to the hidden markov model defined by the 
    transition probabilities in A and the emission probabilites in B. do so 
    using the forward algorithm. return the matrix of the log of the forward-
    path probabilities, if desired
    - input:
      - O:              the observation sequence under consideration
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - pi:             the vector of the initial-state probabilities
      - V:              the vocabulary of ordered possible observations
      - return_matrix:  return the matrix of forward-path probabilities (ln_alpha)
    - output:
      - ln_P_O:            the likelihood of the given sequence of observations
      - ln_alpha:          N-by-T matrix of the log of that forward-path probs
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # create a matrix to hold the logs of the forward path probabilities
    ln_alpha = np.zeros((N,T))
    # reshape the initial-state probabilities if it's not a 1D array
    if len(pi.shape) > 1:
        pi = pi.squeeze()
    # fill in the first column of matrix with the initial values for the 
    # forward path probabilties
    first_o_index = V.index(O[0])
    ln_alpha[:,0] = np.log(pi) + np.log(B[:,first_o_index])
    # using recursion, fill in the rest of the logs of the forward-path probs
    for j in range(1,T):
        # find the index of the observation at time step j
        o_index = V.index(O[j])
        # walk down the j-th column
        for i in range(N):
            # compute and store the logs of the probabilities of getting to
            # state i at time j from each possible state at time j-1 (ignore 
            # the probability of seeing the observation at time j, for now, but
            # do consider the probability associated with seeing the sequence 
            # of observations up until time j-1)
            ln_prob_from_previous = ln_alpha[:,j-1] + np.log(A[:,i])
            # pull out the maximum value of these log probabilities
            max_ln_prob = np.max(ln_prob_from_previous)
            # apply the log-sum-exp trick. ln_alpha[i,j] is the natural log of 
            # the joint probability of being in the i-th state at the j-th time 
            # step and seeing the first j observations
            ln_alpha[i,j] = np.log(B[i,o_index]) + max_ln_prob + \
                      np.log(np.sum(np.exp(ln_prob_from_previous-max_ln_prob)))
    # the natural log of the probability of the entire sequence is found by 
    # exponentiating and then summing down the last column of the matrix of the
    # natural logs of the forward-path probabilities
    ln_P_O = np.log(np.sum(np.exp(ln_alpha[:,-1])))
    # return the log of the probability of the observation sequence and, if 
    # desired, the matrix of natural logs of the forward-path probabilities
    if return_matrix:
        return ln_P_O, ln_alpha
    else:
        return ln_P_O
#-----------------------------------------------------------------------------#
def forward(O, A, B, pi, V, return_matrix=False):
    '''
    compute the log likelihood (i.e. the natural log of the probability) of the 
    given sequence, O, with respect to the hidden markov model defined by the 
    transition probabilities in A and the emission probabilites in B. do so 
    using the forward algorithm and the extended subroutines suggested by mann, 
    algorithm 5. return the matrix of the log of the forward-path 
    probabilities, if desired
    - input:
      - O:              the observation sequence under consideration
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - pi:             the vector of the initial-state probabilities
      - V:              the vocabulary of ordered possible observations
      - return_matrix:  return matrix of forward-path probabilities (ln_alpha)
    - output:
      - ln_P_O:            the likelihood of the given sequence of observations
      - ln_alpha:          N-by-T matrix of the log of that forward-path probs
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # create a matrix to hold the logs of the forward path probabilities, 
    # initialize the matrix to hold all nans
    ln_alpha = np.nan*np.ones((N,T))
    # reshape the initial-state probabilities if it's not a 1D array
    if len(pi.shape) > 1:
        pi = pi.squeeze()
    # fill in the first column of matrix with the initial values for the 
    # forward path probabilties
    first_o_index = V.index(O[0])
    ln_alpha[:,0] = ln_product(extended_ln(pi), extended_ln(B[:,first_o_index]))
    # using recursion, fill in the rest of the logs of the forward-path probs
    for j in range(1,T):
        # find the index of the observation at time step j
        o_index = V.index(O[j])
        # walk down the j-th column
        for i in range(N):
            # partially compute the log of the alpha value by summing over all 
            # possible previous states. N.B. ln_alpha[i,j] is the log 
            # probability of being in the i-th state at the j-th time step and 
            # seeing the first j observations. the summation only runs over the
            # transition probabilities, so start with that
            for i_previous in range(N):
                ln_alpha[i,j]  = ln_sum(ln_alpha[i,j], 
                                        ln_product(ln_alpha[i_previous,j-1],
                                                   extended_ln(A[i_previous,i])))
            # now, include the probability of the seeing the i-th observation
            ln_alpha[i,j] = ln_product(ln_alpha[i,j], extended_ln(B[i,o_index]))
    # the natural log of the probability of the entire sequence is found by 
    # exponentiating and then summing down the last column of the matrix of the
    # natural logs of the forward-path probabilities
    ln_P_O = extended_ln(np.sum(extended_exp(ln_alpha[:,-1])))
    # return the log of the probability of the observation sequence and, if 
    # desired, the matrix of natural logs of the forward-path probabilities
    if return_matrix:
        return ln_P_O, ln_alpha
    else:
        return ln_P_O
#-----------------------------------------------------------------------------#
def viterbi_basic(O, A, B, pi, V, Q):
    '''
    given a sequence of observations, O, this subroutine finds the most 
    probable sequence of hidden states that could have generated it, based on
    the hidden markov model defined by the transition probabilities, the 
    emission probabilities, B, and the initial-state probabilities, pi. the
    best path is found using the viterbi algorithm.
    - input:
      - O:             the observation sequence under consideration
      - A:             the matrix of transition probabilities
      - B:             the matrix of emission probabilities
      - pi:            the vector of the initial-state probabilities
      - V:             the vocabulary of ordered possible observations
      - Q:             set of possible states
    - output:
      - best_path:     the most likely sequence of hidden states
      - P_best_path:   the probability of the best path
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # initialize matrices to hold viterbi probabilities and backpointers
    viterbi_probs = np.zeros((N,T))
    backpointer = np.zeros((N,T))
    # fill in the first column of the viterbi matrix 
    first_o_index = V.index(O[0])
    viterbi_probs[:,0] = np.multiply(pi,B[:,first_o_index])
    # the first column of the backpointer matrix is not defined, since there is 
    # no previous hidden state. fill with nans
    backpointer[:,0] = np.nan*np.ones(N)
    # using recursion, fill in the rest of the previous viterbi probabilities, 
    # along the way, record the corresponding backpointers
    for j in range(1,T):
        # find the index of the observation at time step j
        o_index = V.index(O[j])
        # walk down the j-th column
        for i in range(N):
            # compute the viterbi probabilty by first computing the probability 
            # of having come from each of the previous states, then taking the 
            # max
            possible_path_probs = np.zeros(N)
            for i_previous in range(N):
                possible_path_probs[i_previous] = viterbi_probs[i_previous,j-1]*A[i_previous,i]*B[i,o_index]
            viterbi_probs[i,j] = np.max(possible_path_probs)
            # the corresponding backpointer index is simply the argmax of the
            # list of computed probabilties
            backpointer[i,j] = np.argmax(possible_path_probs)
    print_matrix(viterbi_probs, name='viterbi_probs')
    print_matrix(extended_ln(viterbi_probs), name='extended_ln(viterbi_probs)')
    # compute the probability of the most-likely path
    P_best_path = np.max(viterbi_probs[:,-1])
    # compute the last backpointer
    final_backpointer = np.argmax(viterbi_probs[:,-1])
    # find the most likely hidden-state sequence by starting with the final 
    # backpointer and working backwards in time (column by column) through the 
    # backpointer matrix to recover the index of the most-likely previous state
    reverse_best_path_indices = [final_backpointer]
    for j in range(T-1,0,-1):
        current_best_index = int(reverse_best_path_indices[-1])
        previous_state_index = int(backpointer[current_best_index,j])
        reverse_best_path_indices.append(previous_state_index)
    # running backwards through the reversed list of best-path indices, make a 
    # list of the best path through the hidden states running forward in time
    best_path = []
    for j in range(T-1,-1,-1):
        best_path.append(Q[reverse_best_path_indices[j]])
    # return the best path and the its likelihood
    return best_path, P_best_path
#-----------------------------------------------------------------------------#
def viterbi_log(O, A, B, pi, V, Q):
    '''
    given a sequence of observations, O, this subroutine finds the most 
    probable sequence of hidden states that could have generated it, based on
    the hidden markov model defined by the transition probabilities, the 
    emission probabilities, B, and the initial-state probabilities, pi. the
    best path is found using the viterbi algorithm. in contrast to 
    viterbi_basic, this subroutine works with the natural logarithms of the 
    viterbi probabilities, thereby avoiding numerical underflow
    - input:
      - O:                  the observation sequence under consideration
      - A:                  the matrix of transition probabilities
      - B:                  the matrix of emission probabilities
      - pi:                 the vector of the initial-state probabilities
      - V:                  the vocabulary of ordered possible observations
      - Q:                  set of possible states
    - output:
      - best_path:          the most likely sequence of hidden states
      - ln_P_best_path:     the natural log of the probability of the best path
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # initialize matrices to hold the natural logarithms of the viterbi 
    # probabilities, phi, and the backpointers
    phi = np.zeros((N,T))
    backpointer = np.zeros((N,T))
    # fill in the first column of the phi matrix 
    first_o_index = V.index(O[0])
    phi[:,0] = np.log(pi) + np.log(B[:,first_o_index])
    # the first column of the backpointer matrix is not defined, since there is 
    # no previous hidden state. fill with nans
    backpointer[:,0] = np.nan*np.ones(N)
    # using recursion, fill in the rest of the natural logs of the viterbi 
    # probabilities, along the way, record the corresponding backpointers
    for j in range(1,T):
        # find the index of the observation at time step j
        o_index = V.index(O[j])
        # walk down the j-th column
        for i in range(N):
            # compute the natural log of the viterbi probabilty by first 
            # computing the natural log of the probability of having come from 
            # each of the previous states, then taking the maximum. the 
            # corresponding backpointer index is simply the argmax of the list 
            # of computed probabilties
            ln_possible_path_probs = np.zeros(N)
            for i_previous in range(N):
                ln_possible_path_probs[i_previous] = phi[i_previous,j-1] +\
                                                     np.log(A[i_previous,i]) +\
                                                     np.log(B[i,o_index])
            phi[i,j] = np.max(ln_possible_path_probs)
            backpointer[i,j] = np.argmax(ln_possible_path_probs)
    # compute the natural log of the probability of the most-likely path and 
    # the last backpointer
    ln_P_best_path = np.max(phi[:,-1])
    final_backpointer = np.argmax(phi[:,-1])
    # find the most likely hidden-state sequence by starting with the final 
    # backpointer and working backwards in time (column by column) through the 
    # backpointer matrix to recover the index of the most-likely previous state
    reverse_best_path_indices = [final_backpointer]
    for j in range(T-1,0,-1):
        current_best_index = int(reverse_best_path_indices[-1])
        previous_state_index = int(backpointer[current_best_index,j])
        reverse_best_path_indices.append(previous_state_index)
    # running backwards through the reversed list of best-path indices, make a 
    # list of the best path through the hidden states running forward in time
    best_path = []
    for j in range(T-1,-1,-1):
        best_path.append(Q[reverse_best_path_indices[j]])
    # return the best path and the its likelihood
    return best_path, ln_P_best_path
#-----------------------------------------------------------------------------#
def viterbi(O, A, B, pi, V, Q):
    '''
    given a sequence of observations, O, this subroutine finds the most 
    probable sequence of hidden states that could have generated it, based on
    the hidden markov model defined by the transition probabilities, the 
    emission probabilities, B, and the initial-state probabilities, pi. the
    best path is found using the viterbi algorithm. in contrast to 
    viterbi_basic, this subroutine works with the natural logarithms of the 
    viterbi probabilities, using the extended logarithmic subroutines of mann,
    thereby avoiding numerical underflow
    - input:
      - O:                  the observation sequence under consideration
      - A:                  the matrix of transition probabilities
      - B:                  the matrix of emission probabilities
      - pi:                 the vector of the initial-state probabilities
      - V:                  the vocabulary of ordered possible observations
      - Q:                  set of possible states
    - output:
      - best_path:          the most likely sequence of hidden states
      - ln_P_best_path:     the natural log of the probability of the best path
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # initialize matrices to hold the natural logarithms of the viterbi 
    # probabilities, phi, and the backpointers
    phi = np.zeros((N,T))
    backpointer = np.zeros((N,T))
    # fill in the first column of the phi matrix 
    first_o_index = V.index(O[0])
    phi[:,0] = ln_product(extended_ln(pi), extended_ln(B[:,first_o_index]))
    # the first column of the backpointer matrix is not defined, since there is 
    # no previous hidden state. fill with nans
    backpointer[:,0] = np.nan*np.ones(N)
    # using recursion, fill in the rest of the natural logs of the viterbi 
    # probabilities, along the way, record the corresponding backpointers
    for j in range(1,T):
        # find the index of the observation at time step j
        o_index = V.index(O[j])
        # walk down the j-th column
        for i in range(N):
            # compute the natural log of the viterbi probabilty by first 
            # computing the natural log of the probability of having come from 
            # each of the previous states, then taking the maximum 
            ln_possible_path_probs = np.zeros(N)
            for i_previous in range(N):
                ln_possible_path_probs[i_previous] = reduce(ln_product, 
                                                         [phi[i_previous,j-1],
                                                          extended_ln(A[i_previous,i]),
                                                          extended_ln(B[i,o_index])])
            # to treat the case where there might be nans in the array of 
            # possible path probabilities, replace them with -inf values. why?
            # because max and np.max both return nan if there's even a single
            # nan in the array. -inf behaves differently
            nan_locs = np.argwhere(np.isnan(ln_possible_path_probs))
            ln_possible_path_probs[nan_locs] = -np.inf
            # pull out the highest possible-path probability
            phi[i,j] = np.max(ln_possible_path_probs)
            # the corresponding backpointer index is simply the argmax of the 
            # list of computed probabilties
            backpointer[i,j] = np.argmax(ln_possible_path_probs)
    # compute the natural log of the probability of the most-likely path and 
    # the last backpointer
    ln_P_best_path = np.max(phi[:,-1])
    final_backpointer = np.argmax(phi[:,-1])
    # find the most likely hidden-state sequence by starting with the final 
    # backpointer and working backwards in time (column by column) through the 
    # backpointer matrix to recover the index of the most-likely previous state
    reverse_best_path_indices = [final_backpointer]
    for j in range(T-1,0,-1):
        current_best_index = int(reverse_best_path_indices[-1])
        previous_state_index = int(backpointer[current_best_index,j])
        reverse_best_path_indices.append(previous_state_index)
    # running backwards through the reversed list of best-path indices, make a 
    # list of the best path through the hidden states running forward in time
    best_path = []
    for j in range(T-1,-1,-1):
        best_path.append(Q[reverse_best_path_indices[j]])
    # return the best path and the its likelihood
    return best_path, ln_P_best_path
#-----------------------------------------------------------------------------#
def backward_basic(O, A, B, pi, V, return_matrix=False):
    '''
    compute the likelihood (i.e. the probability) of the given sequence, O, 
    with respect to the hidden markov model defined by the transition
    probabilities in A and the emission probabilites in B. do so using the 
    backward algorithm. return the backward probabilities, if desired
    - input:
      - O:              the observation sequence under consideration
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - pi:             the vector of the initial-state probabilities
      - V:              the vocabulary of ordered possible observations
      - return_matrix:  return the matrix of backward probabilities (beta)
    - output:
      - P_O:            the likelihood of the given sequence of observations
      - beta:           N-by-T matrix of backward probabilities
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # create a matrix to hold the backward probabilities
    beta = np.zeros((N,T))
    # initialize the last column of the matrix of backward probabilties. N.B. 
    # the backward probability beta[i,j] is the probability of seeing the 
    # observation sequence o[j+1], o[j+2],..., o[T-1] given that you are in 
    # state i at time j
    beta[:,-1] = np.ones(N)
    # working backwards using recursion, fill in the rest of the backward probs
    for j in range(T-2,-1,-1):
        # find the index of the observation at time step j+1 (next time step)
        o_next_index = V.index(O[j+1])
        # walk down the j-th column
        for i in range(N):
            # compute the beta value by summing over all possible next states
            # N.B. beta[i,j] is the probability of seeing the observation 
            # sequence o[j+1], o[j+2],..., o[T-1] given that you are in state i 
            # at time j
            for i_next in range(N):
                beta[i,j] += A[i,i_next]*B[i_next,o_next_index]*beta[i_next,j+1]
    # the probability of the entire sequence is found by propagating the values
    # in the first column of beta back to the beginning of the sequence using 
    # the initial probability distribution over the states and the emission
    # probabilities
    P_O = 0.0
    first_o_index = V.index(O[0])
    for i in range(N):
        P_O += pi[i]*B[i,first_o_index]*beta[i,0]
    # return the probability of the observation sequence and, if desired, the
    # matrix of backward probabilities
    if return_matrix:
        return P_O, beta
    else:
        return P_O
#-----------------------------------------------------------------------------#
def backward_log(O, A, B, pi, V, return_matrix=False):
    '''
    compute the log likelihood (i.e. the natural log of the probability) of the 
    given sequence, O, with respect to the hidden markov model defined by the 
    transition probabilities in A, the emission probabilites in B, and the 
    initial-state probabilities in pi. do so using the backward algorithm. 
    return the backward log probabilities too, if desired
    - input:
      - O:              the observation sequence under consideration
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - pi:             the vector of the initial-state probabilities
      - V:              the vocabulary of ordered possible observations
      - return_matrix:  return the matrix of backward probabilities (beta)
    - output:
      - ln_P_O:         the log likelihood of the sequence of observations
      - ln_beta:        N-by-T matrix of backward log probabilities
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # create a matrix to hold the natural logs of the backward probabilities.
    # initializing with zeros here automatically sets the last column of the 
    # matrix of backward log probabilties. N.B. the backward log probability 
    # ln_beta[i,j] is the natural logarithm of the probability of seeing the 
    # observation sequence o[j+1], o[j+2],..., o[T-1] given that you are in 
    # state i at time j. the backward probabilities at the last time step are 
    # all one, since the probability of seeing nothing afterwards is certain, 
    # irrespective of final state. the natural log of one, is zero. so, the 
    # final row has already been filled
    ln_beta = np.zeros((N,T))
    # using recursion, backfill the rest of the backward log probabilities
    for j in range(T-2,-1,-1):
        # find the index of the observation at time step j+1 (next time step)
        o_next_index = V.index(O[j+1])
        # walk down the j-th column
        for i in range(N):
            # compute and store the log probabilities of going from state i at
            # time j to each of the N possible states at time j+1, emitting the
            # (j+1)th observation, as well as all the subsequent observations
            # after that
            ln_probs_to_next = np.log(A[i,:].T) + np.log(B[:,o_next_index]) + \
                               ln_beta[:,j+1]
            # applying the log-sum-exp trick, compute the max log-prob value
            max_ln_prob = np.max(ln_probs_to_next)
            # apply the log-sum-exp trick. ln_beta[i,j] is the natural log of 
            # the joint probability of seeing the observation sequence o[j+1], 
            # o[j+2],..., o[T-1] given that you are in state i at time j
            ln_beta[i,j] = max_ln_prob + \
                           np.log(np.sum(np.exp(ln_probs_to_next-max_ln_prob)))
    # the log probability of the entire sequence is found by propagating the 
    # values in the first column of ln_beta back to the beginning of the 
    # sequence using the initial probability distribution over the states and 
    # the emission probabilities
    first_o_index = V.index(O[0])
    ln_P_O = np.log(np.sum(np.exp(np.log(pi)+np.log(B[:,first_o_index])+ln_beta[:,0])))
    # return the log probability of the observation sequence and, if desired, 
    # the matrix of backward log probabilities
    if return_matrix:
        return ln_P_O, ln_beta
    else:
        return ln_P_O
#-----------------------------------------------------------------------------#
def backward(O, A, B, pi, V, return_matrix=False):
    '''
    using algorithm 6 in mann, compute the log likelihood (i.e. the natural log 
    of the probability) of the given sequence, O, with respect to the hidden 
    markov model defined by the transition probabilities in A, the emission 
    probabilites in B, and the initial-state probabilities in pi. do so using 
    the backward algorithm. return the backward log probabilities, if desired
    - input:
      - O:              the observation sequence under consideration
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - pi:             the vector of the initial-state probabilities
      - V:              the vocabulary of ordered possible observations
      - return_matrix:  return the matrix of backward probabilities (beta)
    - output:
      - ln_P_O:         the log likelihood of the sequence of observations
      - ln_beta:        N-by-T matrix of backward log probabilities
    '''
    # extract the number of states and the number of time steps
    N = A.shape[0]
    T = len(O)
    # create a matrix to hold the natural logs of the backward probabilities.
    # initializing with zeros here automatically sets the last column of the 
    # matrix of backward log probabilties. N.B. the backward log probability 
    # ln_beta[i,j] is the natural logarithm of the probability of seeing the 
    # observation sequence o[j+1], o[j+2],..., o[T-1] given that you are in 
    # state i at time j. the backward probabilities at the last time step are 
    # all one, since the probability of seeing nothing afterwards is certain, 
    # irrespective of final state. the natural log of one, is zero. so, the 
    # final row has already been filled
    ln_beta = np.zeros((N,T))
    # using recursion, backfill the rest of the backward log probabilities
    for j in range(T-2,-1,-1):
        # find the index of the observation at time step j+1 (next time step)
        o_next_index = V.index(O[j+1])
        # walk down the j-th column
        for i in range(N):
            # initialize a list to hold the probabilities of moving to each of
            # the hidden states at the next time instance
            ln_probs_to_next = []
            # run though the hidden states that can be transitioned to
            for i_next in range(N):
                # compute the probability of moving to the particular next 
                # hidden state and observing all the subsequent observations
                ln_prob_to_next = reduce(ln_product, [extended_ln(A[i,i_next]), 
                                                      ln_beta[i_next,j+1], 
                                                      extended_ln(B[i_next,o_next_index])])
                # store that probability                
                ln_probs_to_next.append(ln_prob_to_next)
            # now, compute the log of the beta value by summing over the log
            # probabilities of transitioning to each of the next possible 
            # hidden states and seeing all the subsequent observations. N.B. 
            # beta[i,j] is the probability of seeing the observation sequence 
            # o[j+1], o[j+2],..., o[T-1] given that you're in state i at time j
            ln_beta[i,j] = reduce(ln_sum, ln_probs_to_next)
    # the log probability of the entire sequence is found by propagating the 
    # values in the first column of ln_beta back to the beginning of the 
    # sequence using the initial probability distribution over the states and 
    # the emission probabilities
    first_o_index = V.index(O[0])
    ln_P_O = extended_ln(np.sum(pi*B[:,first_o_index]*extended_exp(ln_beta[:,0])))
    # return the log probability of the observation sequence and, if desired, 
    # the matrix of backward log probabilities
    if return_matrix:
        return ln_P_O, ln_beta
    else:
        return ln_P_O
#-----------------------------------------------------------------------------#
def check_alpha_beta(alpha, beta):
    '''
    this function checks to make sure the matrix alpha (holding the forward
    probabilities) and the matrix beta (holding the backward probabilties) have
    been computed correctly. this is checked T separate times. how? because 
    when both the forward and backward probabilities are known for each state
    at a given time instance, the probability of the entire observation 
    sequence (assuming it passes through state i) can be computed. adding up 
    the product of these probabilities across all states at a given time step 
    yields the likelihood of the entire observation sequence, P(O). since there 
    are T time steps in the sequence, P(O) can be computed T times. return a 
    boolean stating whether or not the matrices have been computed correctly
    '''
    # compute the likelihood of the entire observation sequence at each time
    P_O_values = np.sum(np.multiply(alpha, beta), axis=0)
    # compute the discrepancies with respect to the first P(O) value
    discrepancies = P_O_values - P_O_values[0]
    # check to make sure all the discrepancies are zero
    if abs(np.sum(discrepancies)) < len(discrepancies)*np.finfo('float').eps:
        matrices_correct = True
    else:
        matrices_correct = False
    # return the boolean value
    return matrices_correct
#-----------------------------------------------------------------------------#
def check_ln_alpha_ln_beta_log(ln_alpha, ln_beta):
    '''
    this function checks to make sure the matrix ln_alpha (holding the forward
    log probabilities) and the matrix ln_beta (holding the backward log 
    probabilties) have been computed correctly. this is checked T separate 
    times. how? because when both the forward and backward log probabilities 
    are known for each state at a given time instance, the log probability of 
    the entire observation sequence (assuming it passes through state i) can be 
    computed. the log likelihood of the entire observation sequence, ln(P(O)),
    can be found by adding the two matrices together and then using the log-
    sum-exp trick. since there are T time steps in the sequence, ln(P(O)) can 
    be computed T times. all T values should be the same. return a boolean 
    stating whether or not the matrices have been computed correctly
    '''
    # add together the corresponding forward and backward log probabilities
    ln_P_O_and_i = ln_alpha + ln_beta
    # compute the max log probability per time step
    max_ln_P_O_per_t = np.max(ln_P_O_and_i, axis=0)
    # using the log-sum-exp trick, compute the log likelihood of the sequence 
    # of observations at each time step
    ln_P_O_values = max_ln_P_O_per_t + \
                np.log(np.sum(np.exp(ln_P_O_and_i-max_ln_P_O_per_t), axis=0))
    # compute the discrepancies with respect to the first ln(P(O)) value
    discrepancies = ln_P_O_values - ln_P_O_values[0]
    # check to make sure all the discrepancies are zero
    if abs(np.sum(discrepancies)) < len(discrepancies)**2*np.finfo('float').eps:
        matrices_correct = True
    else:
        matrices_correct = False
    # return the boolean value
    return matrices_correct
#-----------------------------------------------------------------------------#
def check_ln_alpha_ln_beta(ln_alpha, ln_beta):
    '''
    this function uses the extended logarithm subroutines to check whether the
    matrix ln_alpha (holding the forward log probabilities) and the matrix 
    ln_beta (holding the backward log probabilties) have been computed 
    correctly. this is checked T separate times. how? because when both the 
    forward and backward log probabilities are known for each state at a given 
    time instance, the log probability of the entire observation sequence 
    (assuming it passes through state i) can be computed. since there are T 
    time steps in the sequence, ln(P(O)) can be computed T times. all T values 
    should be the same. return a boolean stating whether or not the matrices 
    have been computed correctly
    '''
    # add together the corresponding forward and backward log probabilities
    ln_P_O_and_i = ln_product(ln_alpha, ln_beta)
    # sum down the rows of the matrix, using the reduce function
    ln_P_O_values = reduce(ln_sum, ln_P_O_and_i)
    # compute the discrepancies with respect to the first ln(P(O)) value
    discrepancies = ln_P_O_values - ln_P_O_values[0]
    # check to make sure all the discrepancies are zero
    if abs(np.sum(discrepancies)) < len(discrepancies)**2*np.finfo('float').eps:
        matrices_correct = True
    else:
        matrices_correct = False
    # return the boolean value
    return matrices_correct
#-----------------------------------------------------------------------------#
def compute_P_O(alpha, beta):
    '''
    given the matrices of forward and backward probabilties, this subroutine
    computes and returns the probability of the entire observation sequence.
    ideally, this function will be used after running check_alpha_beta(). n.b.
    this is not the only way to compute this quantity! it can also be found
    while computing the matrices of forward and backward probabilities. this is
    really just a helper function for the baum-welch subroutine
    - input:
      - alpha:  matrix of forward probabilities
      - beta:   matrix of backward probabilities
    - output:
      - P_O:    probability of the observation sequence
    '''
    # compute the T values of P(O) that can be computed (one at each time 
    # step). all T values should be the same. you could just pick one or, in 
    # order to get rid of numerical error, average all T of them
    P_O = np.mean(np.sum(np.multiply(alpha, beta), axis=0))
    # return the mean value
    return P_O
#-----------------------------------------------------------------------------#
def compute_ln_P_O_log(ln_alpha, ln_beta):
    '''
    given the matrices of forward and backward log probabilties, this 
    subroutine computes and returns the log probability of the entire 
    observation sequence. ideally, this function will be used after running 
    check_ln_alpha_ln_beta(). n.b. this is not the only way to compute this 
    quantity! it can also be found while computing the matrices of forward and 
    backward log probabilities. this is really just a helper function for the 
    log-space baum-welch subroutine
    - input:
      - ln_alpha:  matrix of forward log probabilities
      - ln_beta:   matrix of backward log probabilities
    - output:
      - ln_P_O:    log probability of the observation sequence
    '''
    # compute the T values of ln(P(O)) that can be computed (one at each time 
    # step). start by adding together the corresponding forward and backward 
    # log probabilities
    ln_P_O_and_i = ln_alpha + ln_beta
    # compute the max log probability per time step
    max_ln_P_O_per_t = np.max(ln_P_O_and_i, axis=0)
    # using the log-sum-exp trick, compute the log likelihood of the sequence 
    # of observations at each time step
    ln_P_O_values = max_ln_P_O_per_t + \
                np.log(np.sum(np.exp(ln_P_O_and_i-max_ln_P_O_per_t), axis=0))
    # all T values should be the same. you could just pick one or, in order to 
    # get rid of numerical error, average all T of them
    ln_P_O = np.mean(ln_P_O_values)
    # return the mean value
    return ln_P_O
#-----------------------------------------------------------------------------#
def compute_ln_P_O(ln_alpha, ln_beta):
    '''
    given the matrices of forward and backward log probabilties, this 
    subroutine computes and returns the log probability of the entire 
    observation sequence using the logarithmic subroutines of  . ideally, this function will be 
    used after running check_ln_alpha_ln_beta(). n.b. this is not the only way 
    to compute this quantity! it can also be found while computing the matrices 
    of forward and backward log probabilities. this is really just a helper 
    function for the log-space baum-welch subroutine
    - input:
      - ln_alpha:  matrix of forward log probabilities
      - ln_beta:   matrix of backward log probabilities
    - output:
      - ln_P_O:    log probability of the observation sequence
    '''
    # add together the corresponding forward and backward log probabilities
    ln_P_O_and_i = ln_product(ln_alpha, ln_beta)
    # sum down the rows of the matrix, using the reduce function
    ln_P_O_values = reduce(ln_sum, ln_P_O_and_i)
    # compute the mean value of the T identical values
    ln_P_O = np.mean(ln_P_O_values)
    # return the mean value
    return ln_P_O
#-----------------------------------------------------------------------------#
def expected_transition_counts_basic(O, A, B, alpha, beta, V):
    '''
    compute the array of the transition probabilities at each of the first T-1
    time steps. this is done by using the transition probabilities, A, and the 
    emission probabilities, B, along with forward and backward probabilities,
    alpha and beta. the forward and backward probabilities, themselves, have 
    been computed using the current estimates of A and B
    - input:
      - O:          the observation sequence under consideration
      - A:          the matrix of transition probabilities
      - B:          the matrix of emission probabilities
      - alpha:      matrix of forward probabilities
      - beta:       matrix of backward probabilities
      - V:          the vocabulary of ordered possible observations
    - output:
      - xi:         N x N x T array of the expected-state-transition counts
    '''
    # extract the number of states and the number of time steps
    N, T = alpha.shape
    # create a 3D array to hold the expected-state-transition-count matrices
    xi = np.zeros((N,N,T))
    # fill in the array, time slice by time slice. n.b. the N-by-N matrix at 
    # the T-th time slice is all zeros (or all nans). why? because xi[i,j,k] is
    # the probability of transitioning from state i at time t[k] to state j at 
    # time t[k+1], but at time t[k=T] (i.e. at the end of the observation 
    # sequence), there is no time left to transition to. you're stuck in the 
    # state you're in. time's up. game's over. what about all the other N-by-N
    # matrices at the preceding time steps? can anything be said about them?
    # sure. if you add up all the values in any one of those matrices, you
    # should get one. why? because adding up all the values in one of those
    # matrices gives the probability of transitioning from any of the N states
    # at time t to any of the states at time t+1, i.e. it's just the 
    # probability of moving forward in time, which is certain
    for k in range(T-1):
        # pull out the vocabulary index of the next time step's observation
        o_next_index = V.index(O[k+1])
        # create the N-by-N expected transition matrix at this time slice
        for i in range(N):
            for j in range(N):
                # multiply together the four relevant probabilities: (1) the
                # forward probability of being in state i at time t and seeing
                # the first t observations, (2) the probability of 
                # transitioning from the state i to state j (which puts you at 
                # time t+1), (3) the probability seeing the (t+1)-th 
                # observation when in state j, and (4) the backward probability 
                # of seeing the observations from time t+2 to the end if in 
                # state j at time t+1. this results in P(O, q_t=i, q_(t+1)=j)
                xi[i,j,k] = alpha[i,k]*A[i,j]*B[j,o_next_index]*beta[j,k+1]
    # compute the likelihood of the observation sequence
    P_O = compute_P_O(alpha, beta)
    # condition all these probabilities on the likelihood of the observation 
    # sequence. the result is P(q_t=i, q_(t+1)=j | O)
    xi /= P_O
    # return the 3D expected-state-transition-count array
    return xi
#-----------------------------------------------------------------------------#
def expected_transition_counts(O, A, B, ln_alpha, ln_beta, V):
    '''
    compute the array of the transition log probabilities at each of the first 
    T-1 time steps. this is done by using the transition probabilities, A, and 
    the emission probabilities, B, along with forward and backward log 
    probabilities, ln_alpha and ln_beta. do the computations in log space. the 
    forward and backward log probabilities, themselves, have been computed 
    using the current estimates of A and B
    - input:
      - O:          the observation sequence under consideration
      - A:          the matrix of transition probabilities
      - B:          the matrix of emission probabilities
      - ln_alpha:   matrix of forward log probabilities
      - ln_beta:    matrix of backward log probabilities
      - V:          the vocabulary of ordered possible observations
    - output:
      - ln_xi:      N x N x T array of the natural logarithms of the expected-
                    state-transition counts
    '''
    # extract the number of states and the number of time steps
    N, T = alpha.shape
    # create 3D array to hold the ln(expected-state-transition-count) matrices
    ln_xi = np.zeros((N,N,T))
    # fill in the array, time slice by time slice. n.b. the N-by-N matrix at 
    # the T-th time slice is all nans. why? because xi[i,j,k] is the 
    # probability of transitioning from state i at time t[k] to state j at 
    # time t[k+1], but at time t[k=T] (i.e. at the end of the observation 
    # sequence), there is no time left to transition to. you're stuck in the 
    # state you're in. time's up. game's over. so xi[i,j,T] = 0 and ln(0)=nan
    ln_xi[:,:,T-1] = np.nan
    # fill in all the other N-by-N matrices at the preceding time steps
    for k in range(T-1):
        # pull out the vocabulary index of the observation at this time step
        o_next_index = V.index(O[k+1])
        # create the N-by-N expected log transition matrix at this time slice
        for i in range(N):
            for j in range(N):
                # add together the four relevant log probabilities: (1) the
                # forward log probability of being in state i at time t and 
                # seeing the first t observations, (2) the natural logarithm of
                # the probability of transitioning from the state i to state j 
                # (which puts you at time t+1), (3) the natural logarithm of 
                # the probability seeing the (t+1)-th observation when in state 
                # j, and (4) the backward log probability of seeing the 
                # observations from time t+2 to the end if in state j at time 
                # t+1. this results in ln(P(O, q_t=i, q_(t+1)=j))
                ln_xi[i,j,k] = reduce(ln_product, [ln_alpha[i,k], 
                                                   extended_ln(A[i,j]), 
                                                   extended_ln(B[j,o_next_index]),
                                                   ln_beta[j,k+1]])
    # compute the log likelihood of the observation sequence
    ln_P_O = compute_ln_P_O(ln_alpha, ln_beta)
    # divide all these log probabilities by the log likelihood of the 
    # observation sequence. the result is ln(P(q_t=i, q_(t+1)=j | O))
    ln_xi -= ln_P_O
    # return the 3D expected-state-log-transition-count array
    return ln_xi
#-----------------------------------------------------------------------------#
def expected_occupancy_counts_basic(alpha, beta):
    '''
    compute the matrix of probabilites, where the i,j-th element gives the
    likelihood of being in state i at the j-th time step. this can be done 
    using the forward and backward probabilities alone
    - input:
      - alpha:      matrix of forward probabilities
      - beta:       matrix of backward probabilities
    - output:
      - gamma:      N x T matrix of the expected-state-occupancy counts
    '''
    # compute the expected-state-occupancy-count matrix, where gamma[i,j] is 
    # the probability of being in state i at the j-th time step, based on the 
    # entire observation sequence, i.e. P(q_t=i | O), where q_t is the state at 
    # time t. start by first computing P(q_t=i, O), which is easier to compute
    # than it looks: it's just the forward probabilties times the backwards 
    # probabilities
    gamma = np.multiply(alpha, beta)
    # compute the likelihood of the observation sequence
    P_O = compute_P_O(alpha, beta)
    # take the joint probability and condition on the probability of the 
    # observation sequence, P(O)
    gamma /= P_O
    # return the expected-state-occupancy-count matrix
    return gamma
#-----------------------------------------------------------------------------#
def expected_occupancy_counts(ln_alpha, ln_beta):
    '''
    compute the matrix of log occupany probabilites, where the i,j-th element 
    contains the log likelihood of being in state i at the j-th time step. this 
    can be done using the forward and backward log probabilities alone
    - input:
      - ln_alpha:      matrix of forward log probabilities
      - ln_beta:       matrix of backward log probabilities
    - output:
      - ln_gamma:      N x T matrix of the log expected-state-occupancy counts
    '''
    # compute the log likelihood of the observation sequence
    ln_P_O = compute_ln_P_O(ln_alpha, ln_beta)
    # compute the log expected-state-occupancy-count matrix, where 
    # ln(gamma)[i,j] is the log probability of being in state i at the j-th 
    # time step, based on the entire observation sequence, i.e. P(q_t=i | O), 
    # where q_t is the state at time t. n.b. since the product of the alpha and
    # beta matrices needs to be divided by P(O). since we're working with 
    # logarithms here, ln(P(O)) needs to be subtracted off from each element
    ln_gamma = ln_product(ln_alpha, ln_beta) - ln_P_O
    # return the log expected-state-occupancy-count matrix
    return ln_gamma
#-----------------------------------------------------------------------------#
def implied_transition_matrix_basic(xi, gamma):
    '''
    given the 3D array of expected state transition counts across the time 
    span, xi, and the 2D array of expected occupancy counts across the time 
    span, gamma, compute the implied likelihoods of transitioning between the 
    states
    '''
    # extract the number of states
    N = xi.shape[0]
    # sum across the T-1 time intervals to find expected value of the number of
    # times a transition between state i and state j occurs
    P_i_to_j = np.sum(xi[:,:,:-1], axis=2)
    # similarly, compute the expected value of the number of times a transition
    # away from state i occurs
    P_from_i = np.sum(gamma[:,:-1], axis=1)
    # convert the result to a column vector, then copy and paste that column
    # until you have an N-by-N matrix
    P_from_i = np.tile(P_from_i.reshape(-1,1), N)
    # divide the two matrices, elementwise. the i,j-th element in the resulting
    # matrix is the implied chance of transitioning from state i to state j
    A_new = P_i_to_j/P_from_i
    # return the computed transition matrix
    return A_new
#-----------------------------------------------------------------------------#
def implied_transition_matrix_log(ln_xi, ln_gamma):
    '''
    given the 3D array of log expected-state transition counts across the time 
    span, ln_xi, and the 2D array of log expected-state-occupancy counts, 
    ln_gamma, compute the implied log likelihoods of transitioning between the 
    states
    '''
    # extract the number of states
    N = ln_xi.shape[0]
    # initialize the new matrix of log transition probabilities
    ln_A_new = np.zeros((N,N))
    # for each element in the transition matrix, use the log-sum-exp trick to 
    # accumulate the log probabilities across the time period of observation
    for i in range(N):
        for j in range(N):
            # pull out the max value across the time span of ln_xi for the 
            # particular transition from state i to state j
            max_ln_xi = np.max(ln_xi[i,j,:-1])
            # similarly, pull out the max value across the time span of 
            # ln_gamma for the particular starting state of the transition 
            # (i.e. starting at state i)
            max_ln_gamma = np.max(ln_gamma[i,:-1])
            # using the log-sum-exp trick, compute implied value of ln(A[i,j,])
            ln_A_new[i,j] = max_ln_xi + \
                            np.log(np.sum(np.exp(ln_xi[i,j,:-1]-max_ln_xi))) -\
                            max_ln_gamma - \
                            np.log(np.sum(np.exp(ln_gamma[i,:-1]-max_ln_gamma)))
    # return the computed implied log transition matrix
    return ln_A_new
#-----------------------------------------------------------------------------#
def implied_transition_matrix(ln_xi, ln_gamma):
    '''
    given the 3D array of log expected-state transition counts across the time 
    span, ln_xi, and the 2D array of log expected-state-occupancy counts, 
    ln_gamma, use the extended-logarithm subroutines to compute the implied log 
    likelihoods of transitioning between the states
    '''
    # extract the number of states
    N, T = ln_gamma.shape
    # sum across the T-1 time intervals to find the log of the expected value 
    # of the number of times a transition between state i and state j occurs.
    # this is an N-by-N matrix
    ln_P_i_to_j = reduce(ln_sum, ln_xi[:,:,:-1].T).T
    # similarly, compute the log of the expected value of the number of times a 
    # transition away from state i occurs. note that this is the same as being
    # in state i, because if you're in state i and you're not at time T, then 
    # that means you're also going to transition away from it. this is an 
    # N-by-1 vector
    ln_P_from_i = reduce(ln_sum, ln_gamma[:,:-1].T)
    # since we're working with logarithms, division becomes subtraction. use
    # transposes of the arrays in order to work with numpy without trouble. the 
    # i,j-th element in the resulting N-by-N matrix is the implied chance of 
    # transitioning from state i to state j
    A_new = (ln_P_i_to_j.T - ln_P_from_i).T
    # return the computed transition matrix
    return A_new
#-----------------------------------------------------------------------------#
def implied_emission_matrix_basic(gamma, O, V):
    '''
    given the matrix of expected state occupancy counts across the time 
    span, gamma, the observation sequence, O, and the ordered vocabulary of 
    possible observations, V, compute the implied likelihood of emitting each 
    of the vocabulary entries from each state
    '''
    # extract the number of states and the number of time points
    N, T = gamma.shape
    # for each state, sum up the expected value of the counts when each of the
    # vocabuary entries is observed
    K = len(V)
    P_o_j_when_i = np.zeros((N,K))
    # run through the states
    for i in range(N):
        # run through the vocabulary
        for j in range(K):
            # sum the expected counts if the j-th vocab entry is seen
            row_sum_selected = 0
            for k in range(T):
                if O[k]==V[j]:
                    row_sum_selected += gamma[i,k]
            P_o_j_when_i[i,j] = row_sum_selected
    # similarly, sum up the expected counts of being in each state
    P_in_i = np.sum(gamma, axis=1)
    # convert the result to a column vector, then copy and paste that column
    # until you have an N-by-len(V) matrix
    P_in_i = np.tile(np.expand_dims(P_in_i, axis=1), K)
    # divide the two matrices, elementwise. the i,j-th element in the resulting
    # matrix is the implied chance of emitting vocab entry j from state i
    B_new = P_o_j_when_i/P_in_i
    # return the computed emission matrix
    return B_new
#-----------------------------------------------------------------------------#
def implied_emission_matrix_log(ln_gamma, O, V):
    '''
    given the matrix of expected log state occupancy counts across the time 
    span, ln_gamma, the observation sequence, O, and the ordered vocabulary of 
    possible observations, V, compute the implied log likelihood of emitting 
    each of the vocabulary entries from each state
    '''
    # extract the number of states and the number of time points
    N = ln_gamma.shape[0]
    # count up the number of possible observation
    K = len(V)
    # convert the list of observations to a numpy array
    O = np.array(O)
    # initialize the new emission matrix
    ln_B_new = np.zeros((N,K))
    # run through the vocabulary (the columns of the emission matrix)
    for j in range(K):
        # for this particular vocabulary observation, pull out the indices of
        # the corresponding column(s) 
        column_indices = O==V[j]
        # if there are no corresponding columns (i.e. the j-th vocabulary entry
        # never appears in the observation sequence), then the emission 
        # probability should be zero. the natural log of zero is nan
        if not np.sum(column_indices):
            ln_B_new[:,j] = np.nan
        else:
            # otherwise, having collected these columns from the expected log-
            # occupancy-counts matrix, lay the groundwork for using the log-
            # sum-exp trick by finding the maximum value of each row
            row_maxes = np.max(ln_gamma[:,column_indices],axis=1).reshape((N,1))
            # start computing the j-th column of the implied log emission 
            # matrix using the log-sum-exp trick (i.e. subtract off the maxes, 
            # exponentiate, sum across the columns, take the natural log, and 
            # add the maxes back)
            ln_B_new[:,j] = np.squeeze(row_maxes + \
                            np.log(np.sum(np.exp(ln_gamma[:,column_indices] - \
                            row_maxes), axis=1)).reshape((N,1)))
    # using the log-sum-exp trick, compute the natural log of the probability 
    # of being in each state
    row_maxes = np.max(ln_gamma, axis=1).reshape((N,1))
    ln_P_in_i = row_maxes + \
                np.log(np.sum(np.exp(ln_gamma-row_maxes),axis=1)).reshape((N,1))
    # from each element in the log emission matrix, subtract off the natural 
    # log of the probability of being in each state
    ln_B_new -= ln_P_in_i
    # return the computed emission matrix
    return ln_B_new
#-----------------------------------------------------------------------------#
def implied_emission_matrix(ln_gamma, O, V):
    '''
    given the matrix of the log of the expected state occupancy counts across 
    the time span, ln_gamma, the observation sequence, O, and the ordered 
    vocabulary of possible observations, V, compute the implied likelihood of 
    emitting each of the vocabulary entries from each state
    '''
    # extract the number of states and the number of time points
    N, T = ln_gamma.shape
    # for each state, sum up the expected value of the counts when each of the
    # vocabuary entries is observed
    K = len(V)
    # initialize the numerator, i.e. the matrix holding the expected counts of 
    # observing the j-th vocabulary entry when in the i-th state
    ln_P_o_j_when_i = np.zeros((N,K))
    # run through the states
    for i in range(N):
        # run through the vocabulary
        for j in range(K):
            # sum the expected counts if the j-th vocab entry is seen
            ln_row_sum_chosen = np.nan
            for t in range(T):
                if O[t]==V[j]:
                    ln_row_sum_chosen = ln_sum(ln_row_sum_chosen,ln_gamma[i,t])
            # store the sum computed
            ln_P_o_j_when_i[i,j] = ln_row_sum_chosen
    # similarly, find the log of the sum of the expected counts of being in 
    # each state. this is an N-by-1 vector
    ln_P_in_i = reduce(ln_sum, ln_gamma.T)
    # divide the matrix by the vector, row-wise. since we're working with logs,
    # subtract instead of divide. transpose the arrays appropriately so that
    # numpy can be used directly. the i,j-th element in the resulting matrix
    # is the implied chance of emitting vocab entry j from state i
    ln_B_new = (ln_P_o_j_when_i.T - ln_P_in_i).T
    # return the computed emission matrix
    return ln_B_new
#-----------------------------------------------------------------------------#
def check_cauchy_convergence(array_old, array_new, convergence_criterion):
    '''
    given the latest estimate of an array along with its previous value and the
    relevant (cauchy) convergence criterion, this function checks to see if the 
    2-norm of the difference between the two arrays is less than the criterion
    '''
    # compute the norm of the difference between the two arrays
    delta_norm = np.linalg.norm(array_new - array_old)
    # compare the norm to the cauchy criterion
    if delta_norm < convergence_criterion:
        converged = True
    else:
        converged = False
    # return the value of the boolean
    return converged
#-----------------------------------------------------------------------------#
def baum_welch_basic(O, n_hidden_states, V, A_init=np.zeros(0), 
                     B_init=np.zeros(0), pi_init=np.zeros(0), 
                     n_max_iterations=1000, verbose=False, 
                     plot_convergence=False, suffix=''):
    '''
    given an observation sequence, O, and initial guesses for the transition 
    matrix, A_init, the emission matrix, B_init, and the initial distribution
    over the states, pi_init, this subroutine implements the baum-welch 
    algorithm (a.k.a. the forward-backward algorithm), which is an expectation-
    maximization (EM) algorithm that iteratively computes the best estimates 
    for the transition matrix, the emission matrix, and the initial-state. as 
    this is the basic version of the subroutine, no logarithmic values are used
    vector
    - input:
      - O:                  the observation sequence under consideration
      - n_hidden_states:    the presumed number of hidden states
      - V:                  the vocabulary of ordered possible observations
      - A_init:             initial guess for the matrix of transition probs
      - B_init:             initial guess for the matrix of emission probs
      - pi_init:            initial guess for the initial-state probabilities
      - n_max_iterations:   the maximum number of EM iterations to try
      - verbose:            print convergence history to the screen
    - output:
      - A_estimate:         implied transition matrix, given O
      - B_estimate:         implied emission matrix, given O
      - pi_estimate:        implied vector of initial-state probs, given O
    '''
    # note the starting time
    start_time = datetime.now()
    # pick the desired style of initialization: 'equal' or 'random', which will
    # be used if no initial guesses have been given
    initialization_type = 'random'
    # count up the number of states and the size of the vocabulary
    N = n_hidden_states
    K = len(V)
    # if no inital guesses given, set initial values of A, B, and pi randomly.
    # remember that the rows of A and B must sum to one, as does the vector pi
    if not A_init.size:
        if initialization_type=='random':
            A_init = np.random.rand(N,N)
            row_sums = np.sum(A_init, axis=1)
            A_init /= row_sums.reshape((N,1))
        if initialization_type=='equal':
            A_init = np.tile(1/N, (N,N))
    if not B_init.size:
        if initialization_type=='random':
            B_init = np.random.rand(N,K)
            row_sums = np.sum(B_init, axis=1)
            B_init /= row_sums.reshape((N,1))
        if initialization_type=='equal':
            B_init = np.tile(np.tile(1/K, K), (N,1))
    if not pi_init.size:
        if initialization_type=='random':
            pi_init = np.random.rand(N,1)
            pi_init /= np.sum(pi_init)
            pi_init = pi_init.reshape((N,1))
        if initialization_type=='equal':
            pi_init = np.tile(1/N, (N,1))    
    # intialize the various variables
    A_new = np.empty((N,N))
    B_new = np.empty((N,K))
    pi_new = np.empty((N,1))
    converged_A = False
    converged_B = False
    if plot_convergence:
        convergence_A = []
        convergence_B = []
    # set the convergence criteria
    machine_eps = np.finfo('float').eps
    cauchy_criterion_A = A_init.size*machine_eps
    cauchy_criterion_B = B_init.size*machine_eps
    # print a header, if desired
    if verbose:
        print('\n  baum-welch convergence:')
    # iterate until convergence or max iterations reached
    for iteration in range(n_max_iterations):
        # set the current estimate for the transition and emission matrices
        if iteration == 0:
            # at the first iteration, use the initial guesses for A, B, and pi
            A_old = A_init
            B_old = B_init
            pi_old = pi_init
            pi_old = pi_old.reshape((N,1))
        else:
            A_old = A_new
            B_old = B_new
            pi_old = pi_new
        # compute the corresponding forward and backward probabilities
        P_O, alpha_old = forward_basic(O, A_old, B_old, pi_old, V, return_matrix=True)
        P_O, beta_old = backward_basic(O, A_old, B_old, pi_old, V, return_matrix=True)
        # expectation step: compute the expected state transition count matrix
        xi = expected_transition_counts_basic(O, A_old, B_old, alpha_old, beta_old, V)
        # expectation step: compute the expected state occupancy count matrix
        gamma = expected_occupancy_counts_basic(alpha_old, beta_old)
        # maximization step: update the estimate for the transition probabilities
        A_new = implied_transition_matrix_basic(xi, gamma)
        # maximization step: update the estimate for the emission probabilities
        B_new = implied_emission_matrix_basic(gamma, O, V)
        # maximization step: back out the new value of the initial distribution
        # from the first column of the expected state occupancy matrix
        pi_new = gamma[:,0]
        # check for convergence of both A and B
        converged_A = check_cauchy_convergence(A_old, A_new, cauchy_criterion_A)
        converged_B = check_cauchy_convergence(B_old, B_new, cauchy_criterion_B)
        # if plotting or printing the convergence, compute the differences
        if plot_convergence or verbose:
            delta_A_norm = np.linalg.norm(A_new - A_old)
            delta_B_norm = np.linalg.norm(B_new - B_old)
        # if plotting, then store the differences
        if plot_convergence:
            convergence_A.append(delta_A_norm)
            convergence_B.append(delta_B_norm)
        # if desired, print the convergence progress
        if verbose:
            print('\n\t'+('iteration '+str(iteration+1)+':').center(20) + \
                  ('||delta_A|| = '+str(delta_A_norm)).ljust(40) + \
                  ('||delta_B|| = '+str(delta_B_norm)).ljust(40)) 
        # if both matrices are converged, then stop iterating 
        if converged_A and converged_B:
            break
    # if the matrices have not yet converged, print a message to the screen
    if not converged_A or not converged_B:
        print('\n\tWARNING: either A or B or both are not converged! \n\t' + \
              'increase the maximum number of iterations.')
        # print the current normed error values and the convergence criteria
        print('\n\t\tA matrix:')
        print('\t\t  - current convergence level: '+str(delta_A_norm))
        print('\t\t  - convergence criterion:     '+str(cauchy_criterion_A))
        print('\n\t\tB matrix:')
        print('\t\t  - current convergence level: '+str(delta_B_norm))
        print('\t\t  - convergence criterion:     '+str(cauchy_criterion_B)+'\n')
    # assign the values found (converged or not) as the final estimates
    A_estimate = A_new
    B_estimate = B_new
    pi_estimate = pi_new
    # note the ending time
    end_time = datetime.now()
    # compute and, if desired, print the training time
    training_time = end_time - start_time
    if verbose:
        training_time_str = str(training_time).split('.')[0]
        print('\n\tbaum-welch training time: ' + training_time_str)
    # if plotting, plot the convergence histories
    if plot_convergence:
        # preliminaries
        plot_name = 'baum-welch convergence'
        if suffix:
            plot_name += ' - ' + suffix
        auto_open = True
        the_fontsize = 14
        fig = plt.figure(plot_name)
        # stretch the plotting window
        width, height = fig.get_size_inches()
        fig.set_size_inches(1.0*width, 1.5*height, forward=True)
        # plot the convergence of the transition matrix
        plt.subplot(2,1,1)
        iterations = list(range(1,iteration+2))
        plt.semilogy(iterations, convergence_A, 'r.-')
        x_min, x_max = plt.xlim()
        plt.semilogy([x_min, x_max], 2*[cauchy_criterion_A], 'r--', 
                 label='$cauchy \; criterion$')
        plt.xlabel('$k$', fontsize=the_fontsize)
        plt.ylabel('$\|A^{(k)}-A^{(k-1)}\|_2$', fontsize=the_fontsize)
        y_min, y_max = plt.ylim()
        delta_y = y_max - y_min
        plt.ylim(y_min-0.2*delta_y, y_max+0.2*delta_y)
        plt.legend(loc='best')
        # plot the convergence of the emission matrix
        plt.subplot(2,1,2)
        iterations = list(range(1,iteration+2))
        plt.semilogy(iterations, convergence_B, 'b.-')
        x_min, x_max = plt.xlim()
        plt.semilogy([x_min, x_max], 2*[cauchy_criterion_B], 'b--', 
                     label='$cauchy \; criterion$')
        plt.xlabel('$k$', fontsize=the_fontsize)
        plt.ylabel('$\|B^{(k)}-B^{(k-1)}\|_2$', fontsize=the_fontsize)
        y_min, y_max = plt.ylim()
        delta_y = y_max - y_min
        plt.ylim(y_min-0.2*delta_y, y_max+0.2*delta_y)
        plt.legend(loc='best')
        # use tight layout
        plt.tight_layout()
        # save plot and close
        print('\n\t'+'saving final image...', end='')
        file_name = plot_name+'.png'
        plt.savefig(file_name, dpi=300)
        print('figure saved: '+plot_name)
        plt.close(plot_name)
        # open the saved image, if desired
        if auto_open:
            webbrowser.open(file_name)
    # return the converged A and B matrices
    return A_estimate, B_estimate, pi_estimate
#-----------------------------------------------------------------------------#
def baum_welch(O, n_hidden_states, V, A_init=np.zeros(0), B_init=np.zeros(0),
               pi_init=np.zeros(0), n_max_iterations=1000, verbose=False, 
               plot_convergence=False, suffix=''):
    '''
    given an observation sequence, O, and initial guesses for the transition 
    matrix, A_init, the emission matrix, B_init, and the initial distribution
    over the states, pi_init, this subroutine implements the baum-welch 
    algorithm (a.k.a. the forward-backward algorithm), which is an expectation-
    maximization (EM) algorithm that iteratively computes the best estimates 
    for the transition matrix, the emission matrix, and the initial-state 
    vector
    - input:
      - O:                  the observation sequence under consideration
      - n_hidden_states:    the presumed number of hidden states
      - V:                  the vocabulary of ordered possible observations
      - A_init:             initial guess for the matrix of transition probs
      - B_init:             initial guess for the matrix of emission probs
      - pi_init:            initial guess for the initial-state probabilities
      - n_max_iterations:   the maximum number of EM iterations to try
      - verbose:            print convergence history to the screen
    - output:
      - A_estimate:         implied transition matrix, given O
      - B_estimate:         implied emission matrix, given O
      - pi_estimate:        implied vector of initial-state probs, given O
    '''
    # note the starting time
    start_time = datetime.now()
    # pick the desired style of initialization: 'equal' or 'random', which will
    # be used if no initial guesses have been given
    initialization_type = 'random'
    # count up the number of states and the size of the vocabulary
    N = n_hidden_states
    K = len(V)
    # if no inital guesses given, set initial values of A, B, and pi randomly.
    # remember that the rows of A and B must sum to one, as does the vector pi
    if not A_init.size:
        if initialization_type=='random':
            A_init = np.random.rand(N,N)
            row_sums = np.sum(A_init, axis=1)
            A_init /= row_sums.reshape((N,1))
        if initialization_type=='equal':
            A_init = np.tile(1/N, (N,N))
    if not B_init.size:
        if initialization_type=='random':
            B_init = np.random.rand(N,K)
            row_sums = np.sum(B_init, axis=1)
            B_init /= row_sums.reshape((N,1))
        if initialization_type=='equal':
            B_init = np.tile(np.tile(1/K, K), (N,1))
    if not pi_init.size:
        if initialization_type=='random':
            pi_init = np.random.rand(N,1)
            pi_init /= np.sum(pi_init)
            pi_init = pi_init.reshape((N,1))
        if initialization_type=='equal':
            pi_init = np.tile(1/N, (N,1))    
    # intialize the various variables
    A_new = np.empty((N,N))
    B_new = np.empty((N,K))
    pi_new = np.empty((N,1))
    converged_A = False
    converged_B = False
    if plot_convergence:
        convergence_A = []
        convergence_B = []
    # set the convergence criteria
    machine_eps = np.finfo('float').eps
    cauchy_criterion_A = A_init.size**2*machine_eps
    cauchy_criterion_B = B_init.size**2*machine_eps
    # print a header, if desired
    if verbose:
        print('\n  baum-welch convergence:')
    # iterate until convergence or max iterations reached
    for iteration in range(n_max_iterations):
        # set the current estimate for the transition and emission matrices
        if iteration == 0:
            # at the first iteration, use the initial guesses for A, B, and pi
            A_old = A_init
            B_old = B_init
            pi_old = pi_init
            pi_old = pi_old.reshape((N,1))
        else:
            A_old = A_new
            B_old = B_new
            pi_old = pi_new
        # compute the corresponding forward and backward log probabilities
        ln_P_O, ln_alpha_old = forward(O, A_old, B_old, pi_old, V, return_matrix=True)
        ln_P_O, ln_beta_old = backward(O, A_old, B_old, pi_old, V, return_matrix=True)
        # expectation step: compute the log expected-state-transition-count matrix
        ln_xi = expected_transition_counts(O, A_old, B_old, ln_alpha_old, ln_beta_old, V)
        # expectation step: compute the log expected-state-occupancy-count matrix
        ln_gamma = expected_occupancy_counts(ln_alpha_old, ln_beta_old)
        # maximization step: update estimate for the transition probabilities.
        # first, compute the log implied-transition probabilities
        ln_A_new = implied_transition_matrix(ln_xi, ln_gamma)
        # exponentiate the log-scale probabilities
        A_new = extended_exp(ln_A_new)
        # maximization step: update estimate for the emission probabilities,
        # first, compute the log implied-emission probabilities
        ln_B_new = implied_emission_matrix(ln_gamma, O, V)
        # exponentiate the log-scale probabilities
        B_new = extended_exp(ln_B_new)
        # maximization step: back out the new value of the initial distribution
        # from the first column of the log expected-state-occupancy matrix
        ln_pi_new = ln_gamma[:,0]
        # exponentiate the log-scale probabilities
        pi_new = extended_exp(ln_pi_new)
        # check for convergence of both A and B
        converged_A = check_cauchy_convergence(A_old, A_new, cauchy_criterion_A)
        converged_B = check_cauchy_convergence(B_old, B_new, cauchy_criterion_B)
        # if plotting or printing the convergence, compute the differences
        if plot_convergence or verbose:
            delta_A_norm = np.linalg.norm(A_new - A_old)
            delta_B_norm = np.linalg.norm(B_new - B_old)
        # if plotting, then store the differences
        if plot_convergence:
            convergence_A.append(delta_A_norm)
            convergence_B.append(delta_B_norm)
        # if desired, print the convergence progress
        if verbose:
            print('\n\t'+('iteration '+str(iteration+1)+':').center(20) + \
                  ('||delta_A|| = '+str(delta_A_norm)).ljust(40) + \
                  ('||delta_B|| = '+str(delta_B_norm)).ljust(40)) 
        # if both matrices are converged, then stop iterating 
        if converged_A and converged_B:
            break
    # if the matrices have not yet converged, print a message to the screen
    if not converged_A or not converged_B:
        print('\n\tWARNING: either A or B or both are not converged! \n\t' + \
              'increase the maximum number of iterations.')
        # print the current normed error values and the convergence criteria
        print('\n\t\tA matrix:')
        print('\t\t  - current convergence level: '+str(delta_A_norm))
        print('\t\t  - convergence criterion:     '+str(cauchy_criterion_A))
        print('\n\t\tB matrix:')
        print('\t\t  - current convergence level: '+str(delta_B_norm))
        print('\t\t  - convergence criterion:     '+str(cauchy_criterion_B)+'\n')
    # assign the values found (converged or not) as the final estimates
    A_estimate = A_new
    B_estimate = B_new
    pi_estimate = pi_new
    # note the ending time
    end_time = datetime.now()
    # compute and, if desired, print the training time
    training_time = end_time - start_time
    if verbose:
        training_time_str = str(training_time).split('.')[0]
        print('\n\tbaum-welch training time: ' + training_time_str)
    # if plotting, plot the convergence histories
    if plot_convergence:
        # preliminaries
        plot_name = 'baum-welch convergence'
        if suffix:
            plot_name += ' - ' + suffix
        auto_open = True
        the_fontsize = 14
        fig = plt.figure(plot_name)
        # stretch the plotting window
        width, height = fig.get_size_inches()
        fig.set_size_inches(1.0*width, 1.5*height, forward=True)
        # plot the convergence of the transition matrix
        plt.subplot(2,1,1)
        iterations = list(range(1,iteration+2))
        plt.semilogy(iterations, convergence_A, 'r.-')
        x_min, x_max = plt.xlim()
        plt.semilogy([x_min, x_max], 2*[cauchy_criterion_A], 'r--', 
                 label='$cauchy \; criterion$')
        plt.xlabel('$k$', fontsize=the_fontsize)
        plt.ylabel('$\|A^{(k)}-A^{(k-1)}\|_2$', fontsize=the_fontsize)
        y_min, y_max = plt.ylim()
        delta_y = y_max - y_min
        plt.ylim(y_min-0.2*delta_y, y_max+0.2*delta_y)
        plt.legend(loc='best')
        # plot the convergence of the emission matrix
        plt.subplot(2,1,2)
        iterations = list(range(1,iteration+2))
        plt.semilogy(iterations, convergence_B, 'b.-')
        x_min, x_max = plt.xlim()
        plt.semilogy([x_min, x_max], 2*[cauchy_criterion_B], 'b--', 
                     label='$cauchy \; criterion$')
        plt.xlabel('$k$', fontsize=the_fontsize)
        plt.ylabel('$\|B^{(k)}-B^{(k-1)}\|_2$', fontsize=the_fontsize)
        y_min, y_max = plt.ylim()
        delta_y = y_max - y_min
        plt.ylim(y_min-0.2*delta_y, y_max+0.2*delta_y)
        plt.legend(loc='best')
        # use tight layout
        plt.tight_layout()
        # save plot and close
        print('\n\t'+'saving final image...', end='')
        file_name = plot_name+'.png'
        plt.savefig(file_name, dpi=300)
        print('figure saved: '+plot_name)
        plt.close(plot_name)
        # open the saved image, if desired
        if auto_open:
            webbrowser.open(file_name)
    # return the converged A and B matrices
    return A_estimate, B_estimate, pi_estimate
#-----------------------------------------------------------------------------#
def generate_observed_sequence(H, B, V, Q):
    '''
    given a hidden-state sequence, H, generate a possible observation sequence, 
    O, by sampling the vocabulary using the given emission probabilities
    - input:
      - H:      the sequence of hidden states
      - B:      the matrix of emission probabilities
      - V:      the vocabulary of ordered possible observations
      - Q:      set of possible states
    - output:
      - O:      the sampled observation sequence
    '''
    # run through the sequence of hidden states and generate observations
    O = [np.random.choice(V, p=B[Q.index(q_t),:]) for q_t in H]
    # return the sampled observation sequence
    return O
#-----------------------------------------------------------------------------#    
def generate_sequence(T, A, B, pi, V, Q):
    '''
    given the parameters of a hidden markov model (A, B, pi) and the sets of 
    possible hidden states and vocabulary of possible observations, this 
    function uses the given probabilties to generate a sequence of hidden 
    states and observations of length T. uses ancestral sampling for a directed
    graph model
    - input:
      - T:      the desired length of the sequence to be generated
      - A:      the matrix of transition probabilities
      - B:      the matrix of emission probabilities
      - pi:     the vector of the initial-state probabilities
      - V:      the vocabulary of ordered possible observations
      - Q:      set of possible states
    - output:
      - H:      the sampled sequence of hidden states
      - O:      the sampled observation sequence
    '''
    # sample the initial distribution to get the first hidden state
    H = [np.random.choice(Q, p=pi)]
    # then, create the rest of the hidden-state sequence
    for _ in range(T-1):
        # extract the index of the previous state
        index_previous_q = Q.index(H[-1])
        # sample the distribution for transitioning away from that state
        H.append(np.random.choice(Q, p=A[index_previous_q,:]))
    # generate a sequence of observations using the emission probabilities
    O = generate_observed_sequence(H, B, V, Q)
    # return the two generated sequences
    return H, O
#-----------------------------------------------------------------------------#
def compute_sequence_accuracy(predicted_sequence, true_sequence):
    '''
    given a true sequence of hidden states and the "best path" of hidden states
    predicted by the viterbi algorithm (corresponding to some oberservation 
    sequence), this function computes the accuracy of the prediction. note that
    this function will work with any two lists of equal length
    '''
    # convert both lists to numpy arrays
    predicted_sequence = np.array(predicted_sequence)
    true_sequence = np.array(true_sequence)
    # count up the number of matching states
    n_matches = np.sum(predicted_sequence==true_sequence)
    # divide the number of matches by the length of the sequences
    accuracy = n_matches/true_sequence.size
    # return the accuracy value
    return accuracy
#-----------------------------------------------------------------------------#
def hidden_sequence_likelihood(H, A, pi, Q):
    '''
    in the rare case where a hidden state sequence, H, is known along with the 
    matrix of transition probabilities, A, from which it was drawn, compute the
    likelihood of that sequence
    - input:
      - H:      the hidden-state sequence to be considered
      - A:      the matrix of transition probabilities
      - Q:      set of possible states
    - ouput:
      - P_H:    the probability of seeing the given hidden-state sequence
    '''
    # pull out the state index of each hidden state
    state_indices = [Q.index(q) for q in H]
    # initialize the probability variable
    P_H = pi[state_indices[0]]
    # run through each transition in the sequence and multiply its probability
    for i in range(len(H)-1):
        # pull out the probability of transitioning to the next state
        P_current_to_next = A[state_indices[i],state_indices[i+1]]
        # multiply it to the running probabilty value
        P_H *= P_current_to_next
    # return the final likelihood value
    return P_H
#-----------------------------------------------------------------------------#
def hidden_sequence_log_likelihood(H, A, pi, Q):
    '''
    in the rare case where a hidden state sequence, H, is known along with the 
    matrix of transition probabilities, A, from which it was drawn, compute the
    log likelihood of that sequence
    - input:
      - H:      the hidden-state sequence to be considered
      - A:      the matrix of transition probabilities
      - Q:      set of possible states
    - ouput:
      - ln_P_H:    the log probability of the given hidden-state sequence
    '''
    # pull out the state index of each hidden state
    state_indices = [Q.index(q) for q in H]
    # initialize the log probability variable
    ln_P_H = np.log(pi[state_indices[0]])
    # run through each transition in the sequence and add its log probability
    for i in range(len(H)-1):
        # pull out the log probability of transitioning to the next state
        ln_P_current_to_next = np.log(A[state_indices[i],state_indices[i+1]])
        # multiply it to the running probabilty value
        ln_P_H += ln_P_current_to_next
    # return the final likelihood value
    return ln_P_H
#-----------------------------------------------------------------------------#
def replace_hidden_state_names(H, old_names, new_names):
    '''
    given a hidden-state sequence, H, replace the state names with new names.
    this function can useful when converting the viterbi path derived from
    baum-welch outputs back to the actual state names. note that this requires
    some user input/intuition; essentially this subroutine can be used to name
    the labeled hidden states produced by baum-welch
    '''
    # convert the hidden-state sequence to a numpy array
    H = np.array(H)
    # run through the list of state names and replace in the sequence
    for i in range(len(old_names)):
        # replace the occurances of this state name
        H[H==old_names[i]] = new_names[i]
    # return the translated hidden-state sequence
    return H
#-----------------------------------------------------------------------------#
def expected_state_persistence(q_current, A, Q):
    '''
    this subroutine applies to markov chains. in the context of a hidden markov
    model, the hidden-state sequence is, by itself, is a markov chain. this
    subroutine answers the following question: if a markov chain is in state
    q_current, then what is the expected number of time steps that this state
    will persist? that is: how long before we switch to a different state?
    - input:
      - q_current:              the current state of the markov chain
      - A:                      the matrix of transition probabilities
      - Q:                      set of possible states
    - ouput:
      - persistence_expected:   the expected number of time steps for which 
                                q_current will persist
    '''
    # pull out the index of current state
    index_current = Q.index(q_current)
    # pull out the probability of staying in the current state
    P_repeat = A[index_current,index_current]
    # implement formula 6b from rabiner for state persistence
    try:
        persistence_expected = 1/(1-P_repeat)
    except RuntimeWarning:
        pass
    # return the expected persistence
    return persistence_expected
#-----------------------------------------------------------------------------#
def forecast_next_observation(O, A, B, pi, V):
    '''
    given an observation sequence, O, with T time steps, this subroutine uses
    the transition and emission probabilties, A and B, to define the 
    probability of seeing each of the vocabulary entries in V at the next time
    step, i.e. o at T+1
    - input:
      - O:              the sampled observation sequence
      - A:              the matrix of transition probabilities
      - B:              the matrix of emission probabilities
      - V:              the vocabulary of ordered possible observations
    - output:
      - p_o_next:       distribution over V for the (T+1)th observation
      - o_most_likely:  based on the distribution, the most-likely o at T+1
    '''
    # compute the matrix of forward probabilities
    P_O, alpha = forward_basic(O, A, B, pi, V, return_matrix=True)
    # compute the likelihood distribution for the next observation
    distribution = reduce(np.matmul, [alpha[:,-1], A, B])/P_O
    # create a dictionary mapping the vocabulary entries to the likelihoods
    p_o_next = dict(zip(V, distribution))
    # pull out the most likely next observation
    index_max_prob = np.argmax(distribution)
    o_most_likely = V[index_max_prob]
    # return the dictionary
    return p_o_next, o_most_likely
#-----------------------------------------------------------------------------#

# hidden markov model definition

# [user input] set of possible hidden states
Q = ['hot', 'cold']
# [user input] vocabulary of possible observations
V = [1, 2, 3]
# [user input] transition probabilites: A[i,j] = the probability of the hidden
# state transitioning from Q[i] to Q[j]
A = [[0.6, 0.4],
     [0.5, 0.5]]
# [user input] emission probabilites: B[i,j] = the probability of observing 
# vocabulary entry V[j] when in hidden state Q[i]
B = [[0.2, 0.4, 0.4],
     [0.5, 0.4, 0.1]]
#B = [[0.0, 0.0, 1.0],  # sanity check: no uncertainty in emissions. baum-welch
#     [1.0, 0.0, 0.0]]  # and viterbi should nail the hidden sequence
#B = [[0.0, 0.1, 0.9],  # sanity check II: slight uncertainty in emissions
#     [0.9, 0.1, 0.0]]
# [user input] initial prob. distribution: pi[i] = the probability of the 
# hidden-state sequence starting in state Q[i]
pi = [0.8, 0.2]
# convert A, B, and pi to numpy arrays
A = np.array(A)
B = np.array(B)
pi = np.array(pi)  


# [user input] specify how long of a sequence to generate
T = 10
# using the given probabilities in A, B, pi, automatically generate a valid 
# hidden-state sequence and a corresponding observation sequence
H, O = generate_sequence(T, A, B, pi, V, Q)



#H = ['hot', 'hot', 'hot', 'cold', 'cold']
#O = [2, 3, 3, 2, 2]
#T = 5



# print the given HMM parameters to the screen
print('\n'+100*'-')
print('\n  HMM parameters:')
print_matrix(A, name='A')
print_matrix(B, name='B')
print_matrix(pi, name='pi')
print('\n\tstates =', Q)
print('\n\tvocabulary =', V)

# task one: compute the likelihood of the given sequence of observations
P_O, alpha = forward_basic(O, A, B, pi, V, return_matrix=True)

print('\n'+100*'-')
print('\n  likelihood computation (basic):')
print('\n\tO =', O)
print_matrix(alpha, name='alpha')
print('\n\tP(O) = %.5f' % P_O)
print('\n'+100*'-')

ln_P_O, ln_alpha = forward_log(O, A, B, pi, V, return_matrix=True)

print('\n  likelihood computation (log space):')
print('\n\tO =', O)
print_matrix(ln_alpha, name='ln_alpha')
print_matrix(extended_exp(ln_alpha), name=' --> alpha')
print('\n\tln(P(O)) = %.5f' % ln_P_O)
print('\n\t --> P(O) = %.6f' % np.exp(ln_P_O))
print('\n'+100*'-')

ln_P_O, ln_alpha = forward(O, A, B, pi, V, return_matrix=True)

print('\n  likelihood computation (log space: mann):')
print('\n\tO =', O)
print_matrix(ln_alpha, name='ln_alpha')
print_matrix(extended_exp(ln_alpha), name=' --> alpha')
print('\n\tln(P(O)) = %.5f' % ln_P_O)
print('\n\t --> P(O) = %.6f' % np.exp(ln_P_O))
print('\n'+100*'-')

# task two: compute the most likely sequence of hidden states for the given 
# sequence of observations
best_path, P_best_path = viterbi_basic(O, A, B, pi, V, Q)

print('\n  decoding (basic):')
print('\n\tbest_path|O =', best_path)
print('\n\tP(best_path|O) = %.6f' % P_best_path)
print('\n'+100*'-')

best_path, ln_P_best_path = viterbi_log(O, A, B, pi, V, Q)

print('\n  decoding (log space):')
print('\n\tbest_path|O =', best_path)
print('\n\tln(P(best_path|O)) = %.6f' % ln_P_best_path)
print('\n\t --> P(best_path|O) = %.6f' % np.exp(ln_P_best_path))
print('\n'+100*'-')

best_path, ln_P_best_path = viterbi(O, A, B, pi, V, Q)

print('\n  decoding (log space: mann):')
print('\n\tbest_path|O =', best_path)
print('\n\tln(P(best_path|O)) = %.6f' % ln_P_best_path)
print('\n\t --> P(best_path|O) = %.6f' % np.exp(ln_P_best_path))
print('\n'+100*'-')

# task three: estimate the parameters of the HMM, i.e. the transition 
# probabilities, A, and the emission probabilities, B, assuming that only the
# observation sequence and the set of possible hidden states are known

# compute the backward probabilites
P_O, beta = backward_basic(O, A, B, pi, V, return_matrix=True)

print('\n  likelihood computation (backward) (basic):')
print_matrix(beta, name='beta')
print('\n\tP(O) = %.5f' % P_O)
print('\n'+100*'-')

ln_P_O, ln_beta = backward_log(O, A, B, pi, V, return_matrix=True)

print('\n  likelihood computation (backward) (log space):')
print_matrix(ln_beta, name='ln_beta')
print_matrix(extended_exp(ln_beta), name=' --> beta')
print('\n\tln(P(O)) = %.5f' % ln_P_O)
print('\n\t --> P(O) = %.6f' % np.exp(ln_P_O))
print('\n'+100*'-')

ln_P_O, ln_beta = backward(O, A, B, pi, V, return_matrix=True)

print('\n  likelihood computation (backward) (log space: mann):')
print_matrix(ln_beta, name='ln_beta')
print_matrix(extended_exp(ln_beta), name=' --> beta')
print('\n\tln(P(O)) = %.5f' % ln_P_O)
print('\n\t --> P(O) = %.6f' % np.exp(ln_P_O))
print('\n'+100*'-')

# check to make sure the forward and backward probabilties are correct
matrices_correct = check_alpha_beta(alpha, beta)

print('\n  alpha and beta correct:', matrices_correct)
print('\n'+100*'-')

matrices_correct = check_ln_alpha_ln_beta(ln_alpha, ln_beta)

print('\n  ln_alpha and ln_beta correct:', matrices_correct)
print('\n'+100*'-')

# compute the 3D arry of expected state-transition counts
xi = expected_transition_counts_basic(O, A, B, alpha, beta, V)

print('\n  expected state-transition counts (basic):')
print_matrix(xi, name='xi')
print('\n'+100*'-')

ln_xi = expected_transition_counts(O, A, B, ln_alpha, ln_beta, V)

print('\n  expected state-transition counts (log space):')
print_matrix(ln_xi, name='ln_xi')
print_matrix(extended_exp(ln_xi), name=' --> xi')
print('\n'+100*'-')

# compute the 2D array of expected occupancy counts
gamma = expected_occupancy_counts_basic(alpha, beta)

print('\n  expected occupancy counts (basic):')
print_matrix(gamma, name='gamma')
print('\n'+100*'-')

ln_gamma = expected_occupancy_counts(ln_alpha, ln_beta)

print('\n  expected occupancy counts (log space):')
print_matrix(ln_gamma, name='ln_gamma')
print_matrix(extended_exp(ln_gamma), name=' --> gamma')
print('\n'+100*'-')

# compute the implied transition matrix, given xi and gamma
A_new = implied_transition_matrix_basic(xi, gamma)

print('\n  implied transition matrix (basic):')
print_matrix(A_new, name='A_new')
print('\n'+100*'-')

ln_A_new = implied_transition_matrix_log(ln_xi, ln_gamma)

print('\n  implied transition matrix (log space):')
print_matrix(ln_A_new, name='ln_A_new')
print_matrix(extended_exp(ln_A_new), name=' --> A_new')
print('\n'+100*'-')

ln_A_new = implied_transition_matrix(ln_xi, ln_gamma)

print('\n  implied transition matrix (log space: mann):')
print_matrix(ln_A_new, name='ln_A_new')
print_matrix(extended_exp(ln_A_new), name=' --> A_new')
print('\n'+100*'-')

# compute the implied emission matrix, given gamma
B_new = implied_emission_matrix_basic(gamma, O, V)

print('\n  implied emission matrix (basic):')
print_matrix(B_new, name='B_new')
print('\n'+100*'-')

ln_B_new = implied_emission_matrix_log(ln_gamma, O, V)

print('\n  implied emission matrix (log space):')
print_matrix(ln_B_new, name='ln_B_new')
print_matrix(extended_exp(ln_B_new), name=' --> B_new')
print('\n'+100*'-')

ln_B_new = implied_emission_matrix(ln_gamma, O, V)

print('\n  implied emission matrix (log space: mann):')
print_matrix(ln_B_new, name='ln_B_new')
print_matrix(extended_exp(ln_B_new), name=' --> B_new')
print('\n'+100*'-')

# compute the implied initial probability distribution, given gamma
pi_new = gamma[:,0]

print('\n  implied initial probabilities (basic):')
print_matrix(pi_new, name='pi_new')
print('\n'+100*'-')

ln_pi_new = ln_gamma[:,0]

print('\n  implied initial probabilities (log space: mann):')
print_matrix(ln_pi_new, name='ln_pi_new')
print_matrix(extended_exp(ln_pi_new), name=' --> pi_new')
print('\n'+100*'-')

# implement the baum-welch algorithm to estimate A and B
A_init = np.random.rand(2,2)
B_init = np.random.rand(2,3)
A_estimate_basic, \
B_estimate_basic, \
pi_estimate_basic = baum_welch_basic(O, 2, V, A_init=A_init, B_init=B_init,
                                     n_max_iterations=500, verbose=True,
                                     plot_convergence=True, suffix='basic')

A_estimate, B_estimate, pi_estimate = baum_welch(O, 2, V, 
                                                 A_init=A_init, B_init=B_init,
                                                 n_max_iterations=500, 
                                                 verbose=True, 
                                                 plot_convergence=True)
print('\n  baum-welch results (basic):')
print_matrix(np.round(A_estimate_basic,3), name='A_estimate')
print_matrix(np.round(B_estimate_basic,3), name='B_estimate')
print_matrix(np.round(pi_estimate_basic,3), name='pi_estimate')
print('\n'+100*'-')
print('\n  baum-welch results (log space):')
print_matrix(np.round(A_estimate,3), name='A_estimate')
print_matrix(np.round(B_estimate,3), name='B_estimate')
print_matrix(np.round(pi_estimate,3), name='pi_estimate')
print('\n'+100*'-')

# reprint the true HMM parameters
print('\n\ttrue parameters:')
print_matrix(np.round(A,3), name='A')
print_matrix(np.round(B,3), name='B')
print_matrix(np.round(pi,3), name='pi')
print('\n\t'+70*'-')

# reprint the observation sequence to the screen
print('\n\tgiven observation sequence:')
print('\n\t\tO =', O)
print('\n\t'+70*'-')

# using the HMM parameters found by baum-welch, run viterbi to figure out the 
# most-likely sequence of states
Q_unknown = ['state #1', 'state #2']
best_path_est, ln_P_best_path_est = viterbi(O, A_estimate, B_estimate, 
                                            pi_estimate, V, Q_unknown)
print('\n\tmost-likely hidden-state sequence, given the learned parameters:')
print('\n\t\tbest_path_estimate|O =', best_path_est)
print('\n\t\tln P(best_path_estimate|O) = %.5f' % ln_P_best_path_est)
print('\n\t\t --> P(best_path_estimate|O) = %.6f' % extended_exp(ln_P_best_path_est))
print('\n\t'+70*'-')

# print the actual generating hidden-state sequence
print('\n\tactual generating sequence:')
print('\n\t\tH =', H)

# replace the generic hidden-state names with specific ones and compute the 
# accuracy of the translated hidden-state sequence
print('\n\t'+70*'-')
print('\n\treplacing hidden-state names:')
new_state_names = ['hot', 'cold']
print_matrix(np.array(list(zip(Q_unknown, new_state_names))), name='  replace w/')
best_path_renamed = replace_hidden_state_names(best_path_est, Q_unknown, new_state_names)
print('\n\t\t --> best_path_estimate|O =', best_path_renamed)
path_accuracy_1 = compute_sequence_accuracy(best_path_renamed, H)
print('\n\t\t\t( accuracy:', np.round(100.0*path_accuracy_1,2), '% )')
expected_persistences = dict(zip(new_state_names, 
                             [round(expected_state_persistence(state, 
                                    A_estimate, new_state_names),4) 
                                    for state in new_state_names]))
print_matrix(expected_persistences, name='expected state persistences', n_indents=3)
print('\n\tOR')
new_state_names = ['cold', 'hot']
print_matrix(np.array(list(zip(Q_unknown, new_state_names))), name='  replace w/')
best_path_renamed = replace_hidden_state_names(best_path_est, Q_unknown, new_state_names)
print('\n\t\t --> best_path_estimate|O =', best_path_renamed)
path_accuracy_2 = compute_sequence_accuracy(best_path_renamed, H)
print('\n\t\t\t( accuracy:', round(100.0*path_accuracy_2,2), '% )')
expected_persistences = dict(zip(new_state_names, 
                             [np.round(expected_state_persistence(state, 
                                    A_estimate, new_state_names),4) 
                                    for state in new_state_names]))
print_matrix(expected_persistences, name='expected state persistences', n_indents=3)
print('\n\t'+70*'-')

# compute and print the probability of getting the hidden-state sequence 100%
# right just by blind guessing
N = len(pi)
n_possible_sequences = N**T
P_perfect_guess = 1.0/n_possible_sequences
print('\n\tchance of getting 100.0 % accuracy by guessing:\t',
      str(round(100.0*P_perfect_guess,4)),'%')

# compute and print the probability of getting the hidden-state sequence
# accuracy found via baum-welch just by blind guessing
path_accuracy = max(path_accuracy_1, path_accuracy_2)
n_correct_steps = round(path_accuracy*T)
n_possible_sequences = N**n_correct_steps
P_as_good_guess = 1.0/n_possible_sequences
print('\n\tchance of getting', str(round(100*path_accuracy,2)), '% accuracy '+\
      'by guessing:\t', str(round(100.0*P_as_good_guess,4)), '%')
print('\n'+100*'-')

# forecast the next observation
p_o_next, o_most_likely = forecast_next_observation(O, A, B, pi, V)
print('\n\tforecasting the next observation in the sequence:')
print('\n\t\tO =', O)
print_matrix(p_o_next, name='P(o_(T+1))', n_indents=2)
print('\n\t\tmost-likely o_(T+1) =', o_most_likely)
print('\n'+100*'-')

'''
# DEMO: use probabilities in A, B, and pi to generate a sequence of length T
T = 5
H_sampled, O_sampled = generate_sequence(T, A, B, pi, V, Q)

print('\n'+'-'*40)
print('\n  DEMO')
print('\n'+'-'*40)

# compute the log likeihood of the generated sequence of hidden states
ln_P_H = hidden_sequence_log_likelihood(H_sampled, A, pi, Q)

# task 1: compute the log likelihood of the generated sequence of observations
ln_P_O = forward(O_sampled, A, B, pi, V)

# task 2: find the most-likely sequence of hidden states, given the generated 
# sequence of observations as well as the log likelihood of this most-likely 
# hidden-state sequence
best_path, ln_P_best_path_given_O = viterbi(O_sampled, A, B, pi, V, Q)

# compute the log likelihood of the predicted, most-likely sequence of observations
ln_P_best_path = hidden_sequence_log_likelihood(best_path, A, pi, Q)

# compute the accuracy of the predicted, most-likely sequence of hidden states
best_path_accuracy = compute_sequence_accuracy(best_path, H_sampled)

# task 3: assume the HMM parameters are not known -- that all all that is known
# is the sequence of observations and the number of possible hidden states. 
# given the sequence of observations, and the presumed number of hidden states, 
# find the best estimates for the HMM parameters: A, B, and pi
A_init = np.random.rand(2,2)
B_init = np.random.rand(2,3)
A_estimate, B_estimate, pi_estimate = baum_welch(O_sampled, 2, V,
                                                 suffix='sampled',
                                                 A_init=A_init, B_init=B_init,
                                                 n_max_iterations=30000, 
                                                 verbose=True,
                                                 plot_convergence=True)

# task 1: based on the parameters found, compute the log likelihood of the 
# sequence of generated observations
ln_P_O_sampled = forward(O_sampled, A_estimate, B_estimate, pi_estimate, V)

# task 2: based on the parameters found, find the most-likely sequence of 
# hidden states, given the generated sequence of observations as well as the 
# log likelihood of this most-likely hidden-state sequence
best_path_sampled, \
ln_P_best_path_given_O_sampled = viterbi(O_sampled, A_estimate, B_estimate, 
                                         pi_estimate, V, Q)

# compute the log likelihood of the predicted, most-likely sequence of observations
ln_P_best_path_sampled = hidden_sequence_log_likelihood(best_path_sampled, 
                                                        A_estimate,
                                                        pi_estimate, Q)

# compute the accuracy of this predicted, most-likely sequence of hidden states
best_path_accuracy_sampled = compute_sequence_accuracy(best_path_sampled, H_sampled)

# print out the results
print('\n  generated sequences:')
print('\n\tH_sampled =', H_sampled)
print('\n\tO_sampled =', O_sampled)

print('\n'+'-'*40)

print('\n  likelihood of the hidden sequence:')
print('\n\tln P(H_sampled) =', ln_P_H)
print('\n\t --> P(H_sampled) =', extended_exp(ln_P_H))

print('\n  likelihood computation:')
print('\n\tln P(O_sampled) =', ln_P_O)
print('\n\t --> P(O_sampled) =', extended_exp(ln_P_O))

print('\n  decoding:')
print('\n\tbest_path|O =', best_path)
print('\n\tln P(best_path|O) =', ln_P_best_path_given_O)
print('\n\t --> P(best_path|O) =', extended_exp(ln_P_best_path_given_O))

print('\n\tln P(best_path) =', ln_P_best_path)
print('\n\t --> P(best_path) =', extended_exp(ln_P_best_path))

print('\n\tbest-path accuracy: %.1f%%' % (100.0*best_path_accuracy))

print('\n'+'-'*40)
                             
print('\n\tbaum-welch results:')
print('\nA_estimate =\n', np.round(A_estimate,2))
print('\nB_estimate =\n', np.round(B_estimate,2))
print('\npi_estimate =\n', np.round(pi_estimate,2))
print('\n\nA_true =\n', np.round(A,2))
print('\nB_true =\n', np.round(B,2))
print('\npi_true =\n', np.round(pi,2))
print('\n'+'-'*40)

print('\n  likelihood computation (based on the parameters found):')
print('\n\tln P(O_sampled) =', ln_P_O_sampled)
print('\n\t --> P(O_sampled) =', extended_exp(ln_P_O_sampled))

print('\n  decoding (based on the parameters found):')
print('\n\tbest_path_sampled =', best_path_sampled)
print('\n\tln P(best_path_sampled|O) =', ln_P_best_path_given_O_sampled)
print('\n\t --> P(best_path_sampled|O) =', extended_exp(ln_P_best_path_given_O_sampled))
print('\n\tln P(best_path_sampled) =', ln_P_best_path_sampled)
print('\n\t --> P(best_path_sampled) =', extended_exp(ln_P_best_path_sampled))
print('\n\tbest-path accuracy: %.1f%%' % (100.0*best_path_accuracy_sampled))

'''

# numba
# monitor P(O) and use for conv. crit.
# prediction of next observation
# expected hidden-state duration (for markov chains)
# training with multiple sequences
# residual checks
# uniform distributions as guesses
# continuous observations distributions
# pomogranate comparison


