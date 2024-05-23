#load the raw .data from the file to give it the correct format
import csv
rutaDatasetOriginal = "./scikit/originalDataset/flare.data2"
rutaCsvResultante = "./scikit/data/flares.csv"

# load the data from the file
with open(rutaDatasetOriginal, "r") as f:
    data = f.read()

# split the data into rows
data = data.split('\n')
data = data[1:-1] # remove header because it is not part of the data and the last empty line
# 
columns = ["index",
           "modifedZurichClass",
           "largestSpotSize",
           "spotDistribution",
           "activity",
           "evolution",
           "previousDailyActivity",
           "historicallyComplex",
           "becameHistoricallyComplex",
           "area",
           "largestSpotArea",
           "cFlares",
           "mFlares",
           "xFlares",]

with open(rutaCsvResultante, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(columns)
    for index, row in enumerate(data):
        row = [index] + row.split(' ')
        csv_writer.writerow(row)
    