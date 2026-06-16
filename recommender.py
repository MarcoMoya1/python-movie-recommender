import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Calculate average rating per movie
avg_ratings = ratings.groupby('movieId')['rating'].mean().round(2)
rating_counts = ratings.groupby('movieId')['rating'].count()

# Merge everything
movies['avg_rating'] = movies['movieId'].map(avg_ratings)
movies['rating_count'] = movies['movieId'].map(rating_counts)
movies = movies.dropna()

# One-hot encode genres
genres_encoded = movies['genres'].str.get_dummies('|')

# Combine genres and rating into one feature set
features = genres_encoded.copy()
features['avg_rating'] = movies['avg_rating'].values

# Calculate similarity between all movies
similarity = cosine_similarity(features)

# Reset index for easy lookup
movies = movies.reset_index(drop=True)

def recommend(movie_title, num_recommendations=5):
    # Find the movie
    matches = movies[movies['title'].str.contains(
                     movie_title, case=False, na=False)]
    
    if matches.empty:
        print(f"Sorry, couldn't find '{movie_title}'")
        return
    
    # Get the first match
    movie_idx = matches.index[0]
    movie = movies.iloc[movie_idx]
    print(f"\nBecause you liked: {movie['title']}")
    print(f"Rating: {movie['avg_rating']} ⭐\n")
    print("We recommend:")
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Print recommendations
    for i, (idx, score) in enumerate(sim_scores, 1):
        m = movies.iloc[idx]
        print(f"{i}. {m['title']} "
              f"(Rating: {m['avg_rating']}⭐, "
              f"Match: {round(score*100)}%)")

# Text interface
print("🎬 Welcome to Python Movie Recommender!")
print("=" * 40)

while True:
    user_input = input("\nEnter a movie name (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        print("Thanks for using Movie Recommender!")
        break
    recommend(user_input)