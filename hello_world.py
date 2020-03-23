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
    rating_count = rating_count.sort_values('rating', ascending=False)

    most_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])

    summary = pd.merge(most_rated_places, cuisine, on='placeID')
    print(summary)

    print(cuisine['Rcuisine'].describe())


if __name__ == "__main__":
    main()
