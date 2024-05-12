import numpy as np

# 1
print("\nEjercicio 1\n")
print(np.arange(0, 101, 4))

# 2
print("\nEjercicio 2\n")
print(np.arange(20).reshape(5, 4))

# 3
print("\nEjercicio 3\n")
array = np.arange(5)
print(array)
print(np.flip(array))

# 4
print("\nEjercicio 4\n")
array = np.arange(9).reshape(3, 3)
print(array)
print(np.flip(array, 0))

# 5
print("\nEjercicio 5\n")
array = np.arange(16, step=2).reshape(2, 4)
print(array)
print(np.mean(array, axis=0))

# 6
print("\nEjercicio 6\n")
array = np.arange(12)
print(array)
print(array.reshape(4, 3)) 

# 7
print("\nEjercicio 7\n")
def transformArrayIntoSquareMatrix(array):
    side = int(array.size ** 0.5)
    if(side * side == array.size):
        print(array.reshape(side, side))
    else:
        print("The array can't be transformed into a square matrix")

array = np.arange(12)
print(array)
transformArrayIntoSquareMatrix(array)
    
array2 = np.arange(16)
print(array2)
transformArrayIntoSquareMatrix(array2)

# 8
print("\nEjercicio 8\n")
array = np.arange(16, step=2).reshape(2, 4)
print(array)
print(np.max(array, axis=1))

# 9
print("\nEjercicio 9\n")
array = np.random.randint(0, 10, 20)
print(array)
print(np.unique(array, return_counts=True))

# 10
print("\nEjercicio 10\n")
array = np.arange(12).reshape(4, 3)
print(array)
mean = np.mean(array, axis=0)
std = np.std(array, axis=0)
print("Media: ", mean)
print("Desv: ", std)
print((array - mean) / std)

# 11
print("\nEjercicio 11\n")
array = np.arange(1, 24, 2).reshape(4, 3)
print(array)
mean = np.mean(array, axis=1, keepdims=True)
std = np.std(array, axis=1, keepdims=True)
print("Media: ", mean)
print("Desv: ", std)
print((array - mean) / std)

# 12
print("\nEjercicio 12\n")
array = np.random.randint(0, 10, 12).reshape(4, 3)
print(array)
print("maximo: ", np.unravel_index(np.argmax(array), array.shape))
print("minimo: ", np.unravel_index(np.argmin(array), array.shape))

# 13
print("\nEjercicio 13\n")
array = np.random.randint(0, 10, 12).reshape(4, 3)
print(array)
orden = np.argsort(array[:, 0])
print(orden)
print(array[orden])

# 14
print("\nEjercicio 14\n")
array = np.random.normal(size=35).reshape(7, 5)
array[array < 0] = 0
print(array)

# 15
print("\nEjercicio 15\n")
array = np.random.randint(0, 100, 10)
print(array)
k = np.random.randint(2, 3)
print(k)
array = np.argsort(array)
print(array[-k:])

# 16
print("\nEjercicio 16\n")
array = np.random.uniform(0, 1, 42).reshape(6, 7)
print(array)
array[:, :2] = 0
array[:, -3:] = 1
print(array)
