"""
this script implements bubble sort
"""
import numpy as np


# --------------------------------------------------------------------------- #
def bubble_sort(numbers):
    """
    given a list or array of numbers, this function will sort the numbers in
    ascending order using bubble sort.
    space: O(1) extra
    time: O(n^2)
    """
    # count up the number of elements in the list
    n = len(numbers)
    # run through the list n-1 times. (you don't have to do all n times,
    # because both the smallest and the second-smallest numbers will have been
    # put in the right places during the penultimate run-through
    for i in range(n-1):
        # during each run-through, look at subsequent pairs of numbers, and put
        # the bigger number in front of the smaller one. so, at the end of the
        # i-th run-through, the maximum value in the list will be at the end.
        # so, during the subsequent run-through, you don't have to consider
        # that max value anymore, and only have to worry about the values in
        # front of it. that is, during the i-th run-through, you only have to
        # worry about the first n-i entries. finally, since a pair is going to
        # comprise a number and the number in front of it, we only have to loop
        # through n-i-1 of the numbers during the i-th run-through. if, during
        # a given run-through, no values have been swapped, then that means the
        # remainder of the list is already in the right order. keep track of
        # this and stop sorting
        values_swapped = False
        for j in range(n-i-1):
            # pull out the j-th value and the one in front of it
            first_value = numbers[j]
            second_value = numbers[j + 1]
            # if the first value happens to be larger, swap their positions
            if first_value > second_value:
                numbers[j] = second_value
                numbers[j + 1] = first_value
                # make a note that a swap was done
                values_swapped = True
        # if no values were swapped during the previous run-through, then exit
        # (n.b. doing this saves a lot of time if the numbers passed in are
        # already mostly sorted)
        if not values_swapped:
            break
    # n.b. you don't need to return the array here, since the array itself is
    # being altered by having been run through this function


# --------------------------------------------------------------------------- #

# make a list of 9 random values between 0 and 100
values = 100 * np.random.randn(9)

# print out the values before sorting
print('\n  bubble sort:')
print('\n\t- original list:\n\t ', values)

# sort the values using bubble sort
bubble_sort(values)

# print out the sorted values
print('\n\t- sorted list:\n\t ', values)
