# Required libraries
# To read and handle data files
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

# For creating vectors from text and determining similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# To create streamlit application
import streamlit as st

# For base64 encoding
import base64

# Supress warnings
import warnings
warnings.filterwarnings('ignore')


# Function to make song links clickable
def make_song_url_clickable(song_link):
    return '<a target="_blank" href={}>Listen on Spotify</a>'.format(song_link)

# Function to make artist links clickable
def make_artist_url_clickable(artist_link):
    return '<a target="_blank" href={}>View on Spotify</a>'.format(artist_link)

# Song recommender
# Read song library data file
song_library = pd.read_csv('App/song_library.csv', na_filter=False)

# Drop "id_artists" field from DataFrame
song_library.drop(['id_artists'], axis=1, inplace=True)

# Reset index for DataFrame
song_library.reset_index(inplace=True, drop=True)

# Create CountVectorizer object to transform text into vector
song_vectorizer = CountVectorizer()

# Fit the vectorizer on "genres" field of song_library DataFrame
song_vectorizer.fit(song_library['genres'])

# Function to recommend more songs based on given song name
def song_recommender(song_name):
    try:
        # Numeric columns (audio features) in song_library DataFrame
        num_cols = ['release_year', 'duration_s', 'popularity', 'danceability', 'energy', 'key', 'loudness',
                    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

        # Create vector from "genres" field (text data) for given song
        text_vec1 = song_vectorizer.transform(song_library[song_library['name']==str(song_name)]['genres']).toarray()

        # Create vector from numerical columns for given song
        num_vec1 = song_library[song_library['name']==str(song_name)][num_cols].to_numpy()

        # Initialise empty list to store similarity scores
        sim_scores=  []

        # For every song/track in song library, determine cosine similarity with given song
        for index, row in song_library.iterrows():
            name = row['name']

            # Create vector from "genres" field for other songs
            text_vec2 = song_vectorizer.transform(song_library[song_library['name']==name]['genres']).toarray()

            # Create vector from numerical columns for other songs
            num_vec2 = song_library[song_library['name']==name][num_cols].to_numpy()

            # Calculate cosine similarity using text vectors
            text_sim = cosine_similarity(text_vec1, text_vec2)[0][0]

            # Calculate cosine similarity using numerical vectors
            num_sim = cosine_similarity(num_vec1, num_vec2)[0][0]

            # Take average of both similarity scores and add to list of similarity scores
            sim = (text_sim + num_sim)/2
            sim_scores.append(sim)
        
        # Add new column containing similarity scores to song_library DataFrame
        song_library['similarity'] = sim_scores

        # Sort DataFrame based on "similarity" column
        song_library.sort_values(by=['similarity', 'popularity', 'release_year'], ascending=[False, False, False], inplace=True)

        # Create DataFrame "recommended_songs" containing 5 songs that are most similar to the given song
        recommended_songs = song_library[['id', 'name', 'artists', 'release_year']][2:7]
        
        # List of URLs for recommended songs
        recommended_songs['Listen on Spotify'] = ['https://open.spotify.com/track/{}'.format(id) for id in recommended_songs['id']]
        recommended_songs['Listen on Spotify'] = recommended_songs['Listen on Spotify'].apply(make_song_url_clickable)
        
        # Drop id column and return recommended_songs DataFrame
        recommended_songs.drop(['id'], axis=1, inplace=True)
        return recommended_songs
    except:
        # If given song is not found in song library then return error message
        error_msg = '{} not found in songs library.'.format(song_name)
        return error_msg


# Artist recommender
# Read artist library data file
artist_library = pd.read_csv('App/artist_library.csv', na_filter=False)

# Reset index for DataFrame
artist_library.reset_index(inplace=True, drop=True)

# Create CountVectorizer object to transform text into vector
artist_vectorizer = CountVectorizer()

# Fit the vectorizer on "genres" field of song_library DataFrame
artist_vectorizer.fit(artist_library['genres'])

# Function to recommend more artists based on given artist name
def artist_recommender(artist_name):
    try:
        # Numeric columns (audio features) in artist_library DataFrame
        num_cols = ['followers', 'popularity']

        # Create vector from "genres" field (text data) for given artist
        text_vec1 = artist_vectorizer.transform(artist_library[artist_library['name']==str(artist_name)]['genres']).toarray()

        # Create vector from numerical columns for given song
        num_vec1 = artist_library[artist_library['name']==str(artist_name)][num_cols].to_numpy()

        # Initialise empty list to store similarity scores
        sim_scores = []

        # For every artist in artist library, determine cosine similarity with given artist
        for index, row in artist_library.iterrows():
            name = row['name']

            # Create vector from "genres" field for other artists
            text_vec2 = artist_vectorizer.transform(artist_library[artist_library['name']==name]['genres']).toarray()

            # Create vector from numerical columns for other songs
            num_vec2 = artist_library[artist_library['name']==name][num_cols].to_numpy()

            # Calculate cosine similarity using text vectors
            text_sim = cosine_similarity(text_vec1, text_vec2)[0][0]

            # Calculate cosine similarity using numerical vectors
            num_sim = cosine_similarity(num_vec1, num_vec2)[0][0]

            # Take average of both similarity scores and add to list of similarity scores
            sim = (text_sim + num_sim)/2
            sim_scores.append(sim)

        # Add new column containing similarity scores to artist_library DataFrame
        artist_library['similarity'] = sim_scores

        # Sort DataFrame based on "similarity" column
        artist_library.sort_values(by=['similarity', 'popularity', 'followers'], ascending=[False, False, False], inplace=True)

        # Create DataFrame "recommended_artists" containing 5 artists that are most similar to the given artist, sort and return this DataFrame
        recommended_artists = artist_library[['id', 'name', 'followers', 'popularity']][2:7]
        recommended_artists.sort_values(by=['popularity', 'followers'], ascending=[False, False], inplace=True)
        
        # List of URLs for recommended artists
        recommended_artists['View on Spotify'] = ['https://open.spotify.com/artist/{}'.format(id) for id in recommended_artists['id']]
        recommended_artists['View on Spotify'] = recommended_artists['View on Spotify'].apply(make_artist_url_clickable)
        
        # Drop id column and return recommended_artists DataFrame
        recommended_artists.drop(['id'], axis=1, inplace=True)
        return recommended_artists
    except:
        # If given artist is not found in artist library then return error message
        error_msg = '{} not found in artists library'.format(artist_name)
        return error_msg


# Complete Spotify recommender
# Function to recommend similar songs and artists based on song name
def spotify_recommender(song_name):
    try:
        # Get DataFrame of recommended songs using song_recommender() function
        recommended_songs = song_recommender(song_name)

        # Create empty DataFrame to store details of recommended artists
        recommended_artists = pd.DataFrame({'name':[], 'followers':[], 'popularity':[]})

        # Get contributing artists for given song
        artists = song_library[song_library['name']==str(song_name)]['artists'].values[0].split(',')

        # For each contributing artist, get recommended artists using artist_recommender() function
        for artist in artists:
            artist_name = artist.strip()

            # Concatenate returned DataFrame with recommended_artists DataFrame
            recommended_artists = pd.concat([recommended_artists, artist_recommender(artist_name)])
        
        # Sort DataFrame based on "popularity" and "followers" columns
        recommended_artists.sort_values(by=['popularity', 'followers'], ascending=[False, False], inplace=True)
        
        # Return recommended songs and artists
        return recommended_songs, recommended_artists.head()
    except:
        # If given song is not found in song library then return error message
        error_msg = '{} not found in songs library.'.format(song_name)
        return [error_msg]
    

# Set app title
st.set_page_config(page_title='Spotify Recommendation System')

# Page background
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
        .stApp {
          background-image: url("data:image/png;base64,%s");
          background-size: cover;
        }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('App/background.png')

# Page title
st.title('Spotify Recommendation System')

# Get input from user
song_name = st.text_input('Song name: ')

if song_name:
    # Call spotify_recommender() function with given song name
    return_values = spotify_recommender(song_name)
    # If there is only 1 return value then display error message
    if len(return_values)==1:
        st.write('')
        st.write('Oops! {} was not found in the songs library.'.format(song_name))
    
    # If return_val1 is not False then display recommended songs and artists
    else:
        recommended_songs = return_values[0]
        recommended_artists = return_values[1]
    
        # Reset index for DataFrames
        recommended_songs.reset_index(drop=True, inplace=True)
        recommended_artists.reset_index(drop=True, inplace=True)
        
        # Rename columns
        recommended_songs.rename(columns={'name':'Name', 'artists':'Artists', 'release_year':'Release Year'}, inplace=True)
        recommended_artists.rename(columns={'name':'Name', 'followers':'Followers', 'popularity':'Popularity'}, inplace=True)
    
        # Display recommended songs
        st.write('')
        st.write('More songs you might like:')
        st.write(recommended_songs.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Display recommended artists
        st.write('')
        st.write('')
        st.write('Other artists you might like:')
        st.write(recommended_artists.to_html(escape=False, index=False), unsafe_allow_html=True)
