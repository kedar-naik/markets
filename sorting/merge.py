"""
this script implements merge sort
"""
import numpy as np


# --------------------------------------------------------------------------- #
def merge_sort(numbers):
    """
    given a list (must be a list) of numbers, this function will sort the
    numbers in ascending order using merge sort. merge sort is a recursive
    "divide and conquer" algorithm. it is good because of its low runtime
    complexity. if randomly indexing into the array is much more expensive
    than sequentially indexing, then this is a good option
    space: O(n) extra
    time: O(n*log(n))
    """
    # this is a recursive algorithm. the base case is when you get down to just
    # two values in either the left half or the right half. you can't split any
    # further and still have it make sense. in order account for the base case,
    # don't do anything to passed-in array unless it's got more than two
    # elements
    if len(numbers) > 1:
        # find the index corresponding to midpoint. if there's an odd number
        # of elements in the array, the first half will be on element longer
        # than the second. n.b. the // operator is floored division,
        # e.g. a//b = int(a/b)
        mid = len(numbers) // 2
        # split the array at the midpoint
        left_half = numbers[:mid]
        right_half = numbers[mid:]
        # recursively call merge_sort on each half
        merge_sort(left_half)
        merge_sort(right_half)
        # empty the numbers list, but make sure it's still pointing to the same
        # spot in memory. that is, don't do numbers=[]
        numbers.clear()
        # run through both sorted left and right halves, checking pairs and
        # putting the smaller values in the merged list along the way
        while len(left_half) > 0 and len(right_half) > 0:
            # compare the first entries in the left-half and right-half lists
            if left_half[0] <= right_half[0]:
                # if the left-half entry is smaller, add it to the sorted list
                numbers.append(left_half[0])
                # and remove that entry from the left-half list altogether
                left_half.pop(0)
            else:
                # if the right-half entry is smaller, add it to the sorted list
                numbers.append(right_half[0])
                # and remove that entry from the left-half list altogether
                right_half.pop(0)
        # now, at this point, the largest value(s) between the two lists
        # will still have not been added to the merged result. since we've been
        # popping off the entries that got added, one of the half lists will be
        # empty, the other will contain the largest values passed in, already
        # sorted in ascending order. figure out whether they are sitting in the
        # left half or the right half and tack them onto the merged result
        if left_half:
            numbers += left_half
        if right_half:
            numbers += right_half
    # n.b. you don't need to return the array here, since the array itself is
    # being altered by having been run through this function


# --------------------------------------------------------------------------- #
def merge_sort_c(numbers):
    """
    given a list or array of numbers, this function will sort the numbers in
    ascending order using merge sort. merge sort is a recursive "divide and
    conquer" algorithm. it is good because of its low runtime complexity. if
    randomly indexing into the array is much more expensive than sequentially
    indexing, then this is a good option
    space: O(n) extra
    time: O(n*log(n))
    the implementation done is more C-like. the pythonic way is in merge_sort()
    """
    # count up the number of elements in the list
    n = len(numbers)
    # this is a recursive algorithm. the base case is when you get down to just
    # two values in either the left half or the right half. you can't split any
    # further and still have it make sense. in order account for the base case,
    # don't do anything to passed-in array unless it's got more than two
    # elements
    if n >= 2:
        # find the index corresponding to midpoint. if there's an odd number
        # of elements in the array, the first half will be on element longer
        # than the second. n.b. the // operator is floored division,
        # e.g. a//b = int(a/b)
        midpoint = n // 2
        # split the array at the midpoint (be sure to work with copies! this is
        # python, not c!)
        left_half = list(numbers[:midpoint])
        right_half = list(numbers[midpoint:])
        # recursively call merge_sort on each half
        merge_sort_c(left_half)
        merge_sort_c(right_half)
        # now, the left half and the right half have been sorted. it is time
        # to merge them. since both lists are already sorted, in order to
        # merge them and have the result be in ascending order, all we have
        # to do is compare pairs from the two lists, moving from left to
        # right. the smaller value from a pair is immediately added to the
        # merged list, starting at the left. start by initializing index
        # variables for each of the three lists, i.e. the left half,
        # the right half, and the merged list
        index_left = 0
        index_right = 0
        index_merged = 0
        # run through both sorted left and right halves, checking pairs and
        # putting the smaller values in the merged list along the way
        while index_left < len(left_half) and index_right < len(right_half):
            # if the left value is smaller, then add it to the merged
            # result. n.b. we can just alter the original array here,
            # since we don't need the numbers that have been stored there
            # anymore
            if left_half[index_left] < right_half[index_right]:
                numbers[index_merged] = left_half[index_left]
                # now, increment index for the left half to move to the next
                # entry
                index_left += 1
            # if the right half's value is smaller, then store it instead
            else:
                numbers[index_merged] = right_half[index_right]
                # as would've been done for the left-half, increment the index
                index_right += 1
            # since we've added a value to the merged list, increment that
            # index
            index_merged += 1
        # now, at this point, the largest value(s) between the two lists
        # will still have not been added to the merged result. figure out
        # whether they are sitting in the left half or the right half and
        # add them to the merged result, one-by-one
        while index_left < len(left_half):
            numbers[index_merged] = left_half[index_left]
            index_left += 1
            index_merged += 1
        while index_right < len(right_half):
            numbers[index_merged] = right_half[index_right]
            index_right += 1
            index_merged += 1
    # n.b. you don't need to return the array here, since the array itself is
    # being altered by having been run through this function


# --------------------------------------------------------------------------- #

# make a list of 9 random values between 0 and 100
values = list(100 * np.random.randn(9))

# print out the values before sorting
print('\n  merge sort:')
print('\n\t- original list:\n\t ', values)

# sort the values using merge sort
# merge_sort_c(values)
merge_sort(values)  # this one requires a list

# print out the sorted values
print('\n\t- sorted list:\n\t ', values)
