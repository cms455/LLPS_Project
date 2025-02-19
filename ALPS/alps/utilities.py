
import numpy as np

def sortNArrays(list_of_arrays, element_dtype=tuple):
    """
    Sort a list of N arrays by the order of the first array in the list,
    ensuring that np.nan values appear at the end.
    e.g. a=[2, 1, np.nan, 3], b=[1, 9, np.nan, 8], c=['a', 'b', np.nan, 'c']
        [a, b, c] -> [(1, 2, 3, np.nan), (9, 1, 8, np.nan), ('b', 'a', 'c', np.nan)]

    Parameters
    ----------
    list_of_arrays: list of lists/1D arrays
        A list containing N lists or 1D arrays to be sorted.
    element_dtype: data type, default: tuple
        The data type of the elements in the returned list.
        Possible values are tuple, list, or np.ndarray.

    Returns
    -------
    list_of_sorted_arrays: list of sorted lists/1D arrays
        The sorted list of arrays, with the specified element data type.
    """

    def is_nan(value):
        """Check if the value is NaN."""
        try:
            return np.isnan(value)
        except TypeError:
            return False

    # Combine the arrays into a list of tuples, with np.nan handled specially
    combined = list(zip(*list_of_arrays))

    # Sort the combined list, treating np.nan as larger than any other value
    combined_sorted = sorted(combined, key=lambda x: (is_nan(x[0]), x[0] if not is_nan(x[0]) else float('inf')))

    # Separate the sorted tuples back into individual arrays
    list_of_sorted_arrays = list(zip(*combined_sorted))

    # Convert the sorted arrays to the desired data type
    if element_dtype == list:
        list_of_sorted_arrays = [list(a) for a in list_of_sorted_arrays]
    elif element_dtype == np.ndarray:
        list_of_sorted_arrays = [np.asarray(a) for a in list_of_sorted_arrays]

    return list_of_sorted_arrays