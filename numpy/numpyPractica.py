import numpy as np

# 1
print(np.arange(0, 101, 4))

# 2
print(np.arange(20).reshape(5, 4))

# 3
array = np.arange(5)
print(array)
print(np.flip(array))

# 4
array = np.arange(9).reshape(3, 3)
print(array)
print(np.flip(array, 0))

# 5
array = np.arange(16, step=2).reshape(2, 4)
print(array)
print(np.mean(array, axis=0))

# 6
array = np.arange(12)
print(array)
print(array.reshape(4, 3))

# 7
array = np.arange(12)
array2 = np.arange(16)
print(array)
print(array2)

def transformArrayIntoSquareMatrix(array):
    side = int(array.size ** 0.5)
    if(side * side == array.size):
        print(array.reshape(side, side))
    else:
        print("The array can't be transformed into a square matrix")
    
transformArrayIntoSquareMatrix(array)
transformArrayIntoSquareMatrix(array2)

# 8
array = np.arange(16, step=2).reshape(2, 4)
print(array)
print(np.max(array, axis=1))

# 9
array = np.random.randint(0, 10, 20)
print(array)
print(np.unique(array, return_counts=True))

# 10


# 11


# 12


# 13


# 14


# 15


# 16

