# Import libraries.
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


def main():
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    frame = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)

    columns = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMBD_url', 'unknown', 'Action',
               'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
    movies = pd.DataFrame(movies)
    movie_names = pd.DataFrame(movies[['item_id', 'movie_title']])
    # print(movie_names.head())

    combined_movies_data = pd.merge(frame, movie_names, on='item_id')
    # print(combined_movies_data.head())

    # print(combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head())

    # Print name of most popular movie.
    print(combined_movies_data[combined_movies_data['item_id'] == 50]['movie_title'].unique())

    # Building a utility matrix.
    rating_crosstab = combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie_title',
                                                       fill_value=0)
    # print(rating_crosstab.head())

    # Transpose utility matrix to prepare for singular-value-decomposition.
    X = rating_crosstab.T

    # Generating the correlation matrix.
    SVD = TruncatedSVD(n_components=12, random_state=17)
    resultant_matrix = SVD.fit_transform(X)
    # print(resultant_matrix.shape)
    corr_mat = np.corrcoef(resultant_matrix)

    # Isolate star wars from correlation matrix.
    movie_names = rating_crosstab.columns
    movies_list = list(movie_names)
    star_wars = movies_list.index('Star Wars (1977)')
    # print(star_wars)

    corr_star_wars = corr_mat[star_wars]
    # print(corr_mat.shape)

    # Recommending a list of highly correlated movies.
    # print(list(movie_names[(corr_star_wars < 1.0) & (corr_star_wars > .9)]))

    # Recommending the most highly correlated movie.
    print(list(movie_names[(corr_star_wars < 1.0) & (corr_star_wars > .95)]))


if __name__ == "__main__":
    main()
