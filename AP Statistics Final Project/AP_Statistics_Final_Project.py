import csv
from collections import deque
import math
import matplotlib.pyplot as plt
# from media import media
import numpy as np
from random import sample

# https://developer.imdb.com/non-commercial-datasets/
dataset, regional = {}, {}
ratings = deque()
for i, file in enumerate(['title.basics.tsv', 'title.ratings.tsv', 'title.akas.tsv']):
    print(file)
    with open(file, 'r', encoding = "utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = '\t', quoting = csv.QUOTE_NONE)
        for isHeader, line in enumerate(csv_reader):
            if isHeader == 0:
                print(line)
            else:
                if i == 0 and line[1] == "movie" and line[7] != '\\N':
                    obj = media(line[0], line[3])
                    obj.runtime, obj.genres = line[7], line[8]
                    dataset[line[0]] = obj
                elif i == 1 and line[1] != '\\N' and line[2] != '\\N' and line[0] in dataset:
                    obj = dataset[line[0]]
                    obj.rating, obj.numVotes = line[1], line[2]
                    ratings.append(line[0])
                elif i == 2 and (line[3] == 'FR' or line[3] == 'ES') and line[0] in ratings:
                    regional.setdefault(line[3], deque()).append(line[0])

y = np.empty(1000)
samples = [sample(regional["FR"], 1000), sample(regional["ES"], 1000), sample(ratings, 1000)]
stats = np.empty([3, 1000])
for i in range(3):
    objects = [dataset[elem] for elem in samples[i]]
    stats[i] = [int(elem.numVotes) for elem in objects]
    if i == 2:
        y = [float(elem.rating) for elem in objects]

    frequencies = {}
    for obj in objects:
        for genre in obj.genres.split(','):
            frequencies[genre] = frequencies.get(genre, 0) + 1
    frequencies = dict(sorted(frequencies.items(), key = lambda item: item[1], reverse = True))
    print()
    print(f"{['France', 'Spain', 'All Regions'][i]}: {frequencies}")
    
    print(f"Median: {np.median(stats[i])}")
    q1, q3 = np.percentile(stats[i], [25, 75])
    iqr = q3 - q1
    print(f"IQR: {iqr}")
    print(f"Lower Outliers: {sum(stats[i] < q1 - 1.5 * iqr)}")
    print(f"Upper Outliers: {sum(stats[i] > q3 + 1.5 * iqr)}")

x = stats[2]
coords = np.array([[x[i], y[i]] for i in range(1000) if q1 - 1.5 * iqr < x[i] < q3 + 1.5 * iqr])
x, y = coords[:,0], coords[:,1]
print()
print(f"Mean of Y: {np.mean(y)}")
print(f"SD of Y: {np.std(y)}")

r = (len(coords) * sum([elem[0] * elem[1] for elem in coords]) - sum(x) * sum(y)) / math.sqrt((len(coords) * sum([elem**2 for elem in x]) - sum(x)**2) * (len(coords) * sum([elem**2 for elem in y]) - sum(y)**2))
b = r * np.std(y) / np.std(x)
a = np.mean(y) - b * np.mean(x)
print()
print(f"r: {r}")
print(f"r^2: {r**2}")
print(f"b: {b}")
print(f"a: {a}")

def f(x):
    return a + b * x

plt.title("France")
plt.hist(stats[0], bins = 20, range = (0, 200), color = 'green', histtype = 'bar', edgecolor = 'black')
plt.xlabel('length of movie in minutes')
plt.ylabel('counts')
plt.show()

plt.title("Spain")
plt.hist(stats[1], bins = 20, range = (0, 200), color = 'blue', histtype = 'bar', edgecolor = 'black')
plt.xlabel('length of movie in minutes')
plt.ylabel('counts')
plt.show()

plt.title("All Regions")
plt.hist(stats[2], bins = 20, range = (0, 200), color = 'red', histtype = 'bar', edgecolor = 'black')
plt.xlabel('length of movie in minutes')
plt.ylabel('counts')
plt.show()

plt.title('Length of Movie vs Rating of Movie')
plt.plot([0, x.max()], [f(0), f(x.max())])
plt.scatter(x, y, color = 'yellow', s = 10)
plt.xlabel('length of movie in minutes')
plt.ylabel('rating of movie')
plt.show()

residuals = (y - [f(elem) for elem in x])
plt.title('Residual Plot')
plt.scatter(x, residuals, color = 'yellow', s = 10)
plt.xlabel('length of movie in minutes')
plt.ylabel('residual')
plt.show()