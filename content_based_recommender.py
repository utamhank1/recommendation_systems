# Import libraries.
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def main():
    cars = pd.read_csv('mtcars.csv')
    cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'weight', 'qsec', 'vs', 'am', 'gear', 'carb']
    # print(cars.head())
    t = [15, 300, 160, 3.2]
    X = cars.iloc[:, [1, 3, 4, 6]].values
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    print(nbrs.kneighbors([t]))

    print(f"The {cars['car_names'][22]} is the most similar car to the specifications the user specified.")


if __name__ == "__main__":
    main()
