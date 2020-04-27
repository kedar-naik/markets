"""
this script implements selection sort
"""
import numpy as np


# -----------------------------------------------------------------------------#
def selection_sort(numbers):
    """
    given a list or array of numbers, this function will sort the numbers in
    ascending order using selection sort. this algorithm is good if the cost of
    doing a swap is high, since there is only one swap per run-through
    space: O(1) extra
    time: O(n^2)
    """
    # count up the number of elements in the list
    n = len(numbers)
    # run through the list
    for i in range(n):
        # in the portion of the list spanning indices i through n-1, find the
        # location of the minimum value. (this is just an argmin operation.)
        # start by initializing the location of the minimum to be i itself
        min_index = i
        # now, run through the numbers in front of i and see if any are smaller
        for j in range(i+1, n):
            # if there's a smaller number than the smallest one already seen,
            # then it's index is the new minimum location
            if numbers[j] < numbers[min_index]:
                min_index = j
        # once the location of the minimum has been found, pull out that value
        # for a second
        min_value = numbers[min_index]
        # put the i-th value where the min value was
        numbers[min_index] = numbers[i]
        # and put the minimum value at index i
        numbers[i] = min_value
    # n.b. you don't need to return the array here, since the array itself is
    # being altered by having been run through this function


# --------------------------------------------------------------------------- #

# make a list of 9 random values between 0 and 100
values = 100 * np.random.randn(9)

# print out the values before sorting
print('\n  selection sort:')
print('\n\t- original list:\n\t ', values)

# sort the values using selection sort
selection_sort(values)

# print out the sorted values
print('\n\t- sorted list:\n\t ', values)
