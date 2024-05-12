import pandas as pd
df = pd.read_csv('pandas/bmw.csv')
print(df.head())

# 1
print("\nEjercicio 1\n")
print(df[:10])

# 2
print("\nEjercicio 2\n")
year = df["year"]
print(year)
print(year.dtype)
print(len(year))

# 3
print("\nEjercicio 3\n")
mileage = df['mileage']
mileage = mileage[0::7]
print(mileage)

# 4
print("\nEjercicio 4\n")
mileage = df['mileage']
mileage = mileage.sample(frac=0.4)
print(mileage)

# 5
print("\nEjercicio 5\n")
mileage = df['mileage']
print(mileage[mileage < 20000])

# 6
print("\nEjercicio 6\n")
mpg = df['mpg']
print(mpg.sort_values(ascending=True))

# 7
print("\nEjercicio 7\n")
engine = df['engineSize']
print("Media: ", engine.mean())
print("Desviacion tipica: ", engine.std())
print("Maximo: ", engine.max())
print("Minimo: ", engine.min())

# 8
print("\nEjercicio 8\n")
print(df.shape, "\n")
print(df.iloc[-3])

# 9
print("\nEjercicio 9\n")
df2 = df[["mileage", "price", "mpg"]]
print(df2)
print(df2.sample(frac=0.2))

# 10
print("\nEjercicio 10\n")
print(df[(df["mileage"] < 10000) & (df["mpg"] > 40)].head)

# 11
print("\nEjercicio 11\n")
df2 = df.copy()
df2["model"].replace({
    " 1 Series": "Series 1",
    " 2 Series": "Series 2",
    " 3 Series": "Series 3",
    " 4 Series": "Series 4",
    " 5 Series": "Series 5",
    " 6 Series": "Series 6",
    " 7 Series": "Series 7",
    " 8 Series": "Series 8",
    " 9 Series": "Series 9",
}, inplace=True) # inplace=True para modificar el dataframe original
print(df2)

# 12
print("\nEjercicio 12\n")
nuevo = {
    "model": "3 Series",
    "year": 2023,
    "price": 22572,
    "transmission": "Automatic",
    "mileage": 74120,
    "fuelType": "Diesel",
    "tax": 160,
    "mpg": 58.4,
    "engineSize": 2.0
}
df2 = df.copy()
df2.loc[len(df)] = nuevo
print(df2.tail())

# 13
print("\nEjercicio 13\n")
array = df.to_numpy()
print(array)
print(type(array))


# 14
print("\nEjercicio 14\n")
df["milesPerYear"] = df["mileage"] / (2024 - df["year"])
print(df["milesPerYear"])
