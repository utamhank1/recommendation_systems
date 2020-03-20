# Import libraries
import numpy as np
import pandas as pd


def main():
    frame = pd.read_csv('rating_final.csv')
    cuisine = pd.read_csv('chefmozcuisine.csv')
    geodata = pd.read_csv('geoplaces2.csv', encoding='latin-1')
    places = geodata[['placeID', 'name']]

    # print(frame.head())
    # print(geodata.head())

    # Grouping and ranking data.
    rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
    # print(rating.head())
    rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
    print(rating.head())
    print(rating.describe())

    print(rating.sort_values('rating_count', ascending=False).head(5))

    print(places[places['placeID'] == 135085])
    print(cuisine[cuisine['placeID'] == 135085])

    # Preparing data for analysis.
    places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
    # print(places_crosstab.head())

    Tortas_ratings = places_crosstab[135085]

    print(Tortas_ratings[Tortas_ratings >= 0])

    # Evaluating similarity based on correlation.
    similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)

    coor_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
    coor_Tortas.dropna(inplace=True)
    print(coor_Tortas.head())

    Tortas_coor_summary = coor_Tortas.join(rating['rating_count'])

    print(Tortas_coor_summary[Tortas_coor_summary['rating_count'] >= 10].sort_values('PearsonR',
                                                                                     ascending=False).head(10))
    places_coor_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index=np.arange(7),
                                      columns=['placeID'])
    summary = pd.merge(places_coor_Tortas, cuisine, on='placeID')
    print(summary)

    print(cuisine['Rcuisine'].describe())

    print(places[places['placeID'] == 135046])


if __name__ == "__main__":
    main()
