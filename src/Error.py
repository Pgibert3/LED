import numpy as np


def ensure_type(name, var, correct_type):
    """
    Ensures var is of type correct_type.
    @param name: name of the variable being type-checked for custom error message
    @type name: str
    @param var: the variable being type-checked
    @type var: any
    @param correct_type: the correct type to check for
    @type correct_type: any
    @raise TypeError: if val is not of type correct_type
    """
    if not isinstance(var, type):
        raise TypeError("% must be of type %, but was of type %."
                        .format(name, correct_type, type(var)))


def validate_arr(arr, correct_shape):
    """
    Ensures arr is of type np.ndarray and has a shape equal to correct_shape
    @param name: name of the arr being checked
    @type name: str
    @param arr: arr being checked
    @type arr: np.ndarray
    @param correct_shape: the correct shape to check for
    @type correct_shape: tuple(2)
    @raise TypeError: if arr is not of type correct_type
    @raise ValueError: if arr is not of shape correct_shape
    """
    ensure_type(name, arr, np.ndarray)
    if np.shape(arr) != correct_shape:
        raise ValueError("% must have a shape of %, but had a shape of %."
                         .format(name, correct_shape, np.shape(arr)))

