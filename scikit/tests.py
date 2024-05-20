#pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
solar_flare = fetch_ucirepo(id=89) 
  
# data (as pandas dataframes) 
X = solar_flare.data.features 
y = solar_flare.data.targets 
  
# metadata 
print(solar_flare.metadata) 
  
# variable information 
print(solar_flare.variables)

csv = solar_flare.data.original.to_csv()
print(csv)
#save to a file
solar_flare.data.original.to_csv('./scikit/importcsv.csv')