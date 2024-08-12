import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
movies = pd.DataFrame({
    'title': [
        'The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions',
        'The Godfather', 'The Godfather Part II', 'The Godfather Part III',
        'The Dark Knight', 'Batman Begins', 'The Dark Knight Rises',
        'Inception', 'Interstellar', 'The Prestige', 'Memento', 'Dunkirk'
    ],
    'genre': [
        'Action, Sci-Fi', 'Action, Sci-Fi', 'Action, Sci-Fi',
        'Crime, Drama', 'Crime, Drama', 'Crime, Drama',
        'Action, Crime, Drama', 'Action, Crime, Drama', 'Action, Crime, Drama',
        'Action, Sci-Fi, Thriller', 'Adventure, Drama, Sci-Fi', 'Drama, Mystery, Sci-Fi',
        'Mystery, Thriller', 'Action, Drama, History'
    ]
})

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, cosine_sim=cosine_sim, num_recommendations=5):
    if title not in movies['title'].values:
        return f"Sorry, '{title}' not found in the movie list."
    
    idx = movies.index[movies['title'] == title].tolist()[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = movies.iloc[movie_indices][['title', 'genre']]
    return recommendations

print("Recommended movies for 'The Matrix':")
print(recommend_movies('The Matrix'))

print("\nRecommended movies for 'Inception':")
print(recommend_movies('Inception'))