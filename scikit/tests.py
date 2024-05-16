import ucimlrepo
from ucimlrepo import fetch_ucirepo

df = fetch_ucirepo(id=89)
print(df.data.features)
print(df.data.targets)
df.data.features.to_csv('data.csv')