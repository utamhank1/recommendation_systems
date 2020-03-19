# Import necessary libraries
import pandas as pd
import numpy as np


def main():
    # Import datasets.
    frame = pd.read_csv('rating_final.csv')
    cuisine = pd.read_csv('chefmozcuisine.csv')

    # View datasets.
    # print(frame.head())
    # print(cuisine.head())

    # Making recommendations based on simple counting.
    rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count())
    print(rating_count.head())


if __name__ == "__main__":
    main()
