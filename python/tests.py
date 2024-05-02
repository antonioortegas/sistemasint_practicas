import numpy as np
import pandas as pd

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
# test some numpy functions
print(np.dot(x, y))
print(np.sum(x))
print(np.mean(x))

# docstrings example
def suma(a, b):
    """returns the sum of two numbers

    Args:
        a (int): first number
        b (int): second number

    Returns:
        int: sum of a and b
    """
    return a + b

a = 5
print(suma(a, 3))