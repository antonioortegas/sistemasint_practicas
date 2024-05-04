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

# 10 / 11
#Normaliza un array bidimensional de 4 filas y 3 columnas restando la media y dividiendo por la desviación típica en cada fila.
array = np.random.randint(0, 10, 12).reshape(4, 3)
print(array)
mean = np.mean(array, axis=1, keepdims=True)
std = np.std(array, axis=1, keepdims=True)
print((array - mean) / std)

# 12
array = np.random.randint(0, 10, 12).reshape(4, 3)
print(array)
print(np.unravel_index(np.argmax(array), array.shape))
print(np.unravel_index(np.argmin(array), array.shape))

# 13
array = np.random.randint(0, 10, 12).reshape(4, 3)
print(array)
orden = np.argsort(array[:, 0])
print(orden)
print(array[orden])

# 14
array = np.random.normal(size=35).reshape(7, 5)
array[array < 0] = 0
print(array)

# 15
array = np.random.randint(0, 100, 32)
print(array)
k = np.random.randint(0, 100)
print(k)
# get all the indexes of the elements that are greater than k
indexes = np.where(array > k)
print(indexes)

# 16
array = np.random.uniform(0, 1, 42).reshape(6, 7)
print(array)
array[:, :2] = 0
array[:, -3:] = 1
print(array)
