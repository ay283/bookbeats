import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.sparse.linalg import svds
import requests
from datetime import datetime, timedelta
from flask import jsonify

# Set the root path for the application, useful for file linking
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory where the current script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define paths to the JSON files containing the datasets for GoodReads and Spotify
goodreads_file_path = os.path.join(current_directory, 'GoodReads_condensed.json')
spotify_file_path = os.path.join(current_directory, 'Spotify_condensed.json')

# Load GoodReads data from JSON and convert it to a pandas DataFrame
with open(goodreads_file_path, 'r') as file:
    book_data = json.load(file)
    book_df = pd.DataFrame.from_dict(book_data)

# Load Spotify data from JSON and convert it to a pandas DataFrame
with open(spotify_file_path, 'r') as file:
    spotify_data = json.load(file)
    spotify_df = pd.DataFrame.from_dict(spotify_data)

# Initialize Flask app with configurations for template reloading and CORS
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

# Mapping of book genres to song genres for initial matching of interests
book_to_song_genres = {'Nonfiction': ['folk', ' indie'], 'Fiction': [' indie', ' pop', 'pop'],
'Fantasy': [' pop', 'electronic', 'pop', ' indie'], 'Romance': [' pop', ' 90s'], 'Historical': [' classic rock', ' blues'],
'History': [' classic rock', ' blues'], 'Childrens': [' pop',  'pop'], 'Cultural': [' 90s', ' 70s'], 
'Sequential Art': ['electronic', 'rock', ' indie'], 'Mystery': ['jazz'], 'Religion': [' indie', 'folk'], 
'Science': ['electronic', 'rock'], 'Science Fiction': [' pop', ' indie rock', 'electronic'], 
'Paranormal': [' classic rock', 'gothic metal', 'hard rock']}

#Get most popular song genres and most popular book genres in our datasets (helping to construct mapping of genres)
def popular_genres():
     #Initialize dictionary of song tags to frequency 
     song_genres_count = {}
     for genre_list in spotify_df['tagstokenized']:
         for genre in genre_list:
             old_count = song_genres_count.get(genre, 0)
             song_genres_count[genre] = old_count+1
    #Initialize dictionary of book genres to frequency 
     book_genres_count = {}
     for genre_list in book_df['genretokenized']:
         for genre in genre_list:
             old_count = book_genres_count.get(genre, 0)
             book_genres_count[genre] = old_count+1
    #Sort unique song tags and unique book genres by highest to lowest frequency
     sorted_song_genre_count = (sorted(song_genres_count.items(), key=lambda x: x[1], reverse= True))
     sorted_book_genres_count = (sorted(book_genres_count.items(), key=lambda x: x[1], reverse= True))
     #Print out the top 20 most popular book genres
     for pair in sorted_book_genres_count[:20]:
             print(pair)
    #Print out the top 20 most popular song tags
     for pair in sorted_song_genre_count[:20]:
             print(pair)

# Function to find the closest related projects using cosine similarity
def closest_projects(project_index_in, project_repr_in, documents, k = 10):
    sims = project_repr_in.dot(project_repr_in[project_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i][0], documents[i][1], documents[i][2],documents[i][4], sims[i]) for i in asort[1:]]
    
def closest_words(word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]] 

# Perform SVD on Spotify data filtered by the book's genres and add our book query for comparison
def svd_calculation(query, filtered_df):
    # Convert filtered DataFrame to JSON and write to a file for processing
    print("start td matrix") 
    json_filtered = filtered_df.to_json(orient= "records")
    
    # Write JSON string to a file
    with open("filtered_data.json", "w") as f:
        f.write(json_filtered)

    # Load JSON data from the file
    with open("filtered_data.json") as f:
        data = json.load(f)

    # Extract text data from JSON records
    documents = [(x['title'], x['artist'], x['spotify_id'],x['text'],x['tagstokenized'])
                for x in data
                if len(x['text'].split()) > 50]
    
    # Ensure that query label will be the last entry in 'docs_compressed_normed' 
    process = [x[3] for x in documents]
    process.append(query) # Add query at the end to preserve order
    print("process")
    vectorizer = CountVectorizer()
    td_matrix = vectorizer.fit_transform(process)
    td_matrix=td_matrix.astype('float64')
    docs_compressed, s, words_compressed= svds(td_matrix, k=100)
    docs_compressed_normed = normalize(docs_compressed)
    
    return docs_compressed_normed, documents

# Filter Spotify data to find songs related to the book's genres
def song_filter(query):
    book = book_df[(book_df['title'] == query)]
    our_genres = book['genretokenized']
    rel_song_genre_set = set()
    for key, s_gen in book_to_song_genres.items():
        for genre in our_genres:
            for genre1 in genre:
                if key == genre1:
                    rel_song_genre_set.update(s_gen)
    if (len(rel_song_genre_set)==0):
        return spotify_df
    rel_song_genre_list = list(rel_song_genre_set)
    print(rel_song_genre_list)
    filtered_df = spotify_df[spotify_df['tagstokenized'].apply(lambda x: any(genre in x for genre in rel_song_genre_list))]
    return filtered_df

@app.route("/")
def home():
    # Pass all book titles to the frontend for user interaction
    all_titles = book_df['title'].tolist()
    return render_template('base.html', title="Sample HTML", all_titles=jsonify(all_titles))

# Route to return top ten similar songs to a book (based on cossine similarity)
@app.route("/songs", methods = ['GET'])
def episodes_search():
    text = request.args.get("title")
    book = book_df[(book_df['title'] == text)]
    if book.empty:
        print("DNE")
        return json.dumps({"error": "Book Not Found"}), 404
   
    # Filter our spotify_df as needed:
    filtered_spotify = song_filter(text)
    song_count = filtered_spotify.shape[0]
    print(song_count, "size of filtered")

    # Perform SVD on Filtered DF 
    desc = return_desc(text)
    print(desc)
    docs_compressed_norm, documents = svd_calculation(desc,filtered_spotify)
    # Provides top ten cossine sim 
    top_ten_songs = closest_projects(song_count, docs_compressed_norm, documents) #Returns tuple list 
    
    # Response json
    response_json = {}
    response_json["top_ten_songs"] = []  
    for title, artist, id, genres, sim in top_ten_songs:
        mapping = {"title": title, "artist": artist, "spotify_id": id, "genres": genres}
        response_json["top_ten_songs"].append(mapping)
        print(title, artist, id, genres, sim)
    return json.dumps(response_json), 200

# Spotify API credentials management
CLIENT_ID = '1124360dd48e4ace9c3be693c3f5f764'
CLIENT_SECRET = '51412712cb8742958a0282f924de6e50'
token_info = {
    'access_token': None,
    'expires_at': None
}

def get_token():
    if not token_info['access_token'] or token_info['expires_at'] < datetime.now():
        # Construct the request for token
        auth_response = requests.post(
            'https://accounts.spotify.com/api/token',
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            data={
                'grant_type': 'client_credentials',
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET
            }
        )
        auth_data = auth_response.json()
        token_info['access_token'] = auth_data['access_token']
        # Set the expiration time
        token_info['expires_at'] = datetime.now() + timedelta(seconds=auth_data['expires_in'])

    return token_info['access_token']

def get_track_info(track_id):
    """ Fetch track information including album data """
    token = get_token()  
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    url = f'https://api.spotify.com/v1/tracks/{track_id}'
    response = requests.get(url, headers=headers)
    track_data = response.json()
    return track_data

@app.route("/album-cover/<spotify_id>")
def get_album_cover(spotify_id):
    print("Spotify id", spotify_id)
    track_data = get_track_info(spotify_id)
    if 'album' in track_data and 'images' in track_data['album'] and track_data['album']['images']:
        album_cover_url = track_data['album']['images'][0]['url']
        return album_cover_url
    else:
        return "Album cover not found"
    
@app.route("/book-titles", methods=['GET'])
def get_book_titles():
    query = request.args.get('query').lower()  # Convert query to lowercase for case-insensitive matching
    # Filter book titles based on the query
    filtered_titles = [title for title in book_df['title'] if query in title.lower()]
    return jsonify(filtered_titles)

def fetch_book_title_suggestions(query):
    # Filter the book titles based on the query
    filtered_titles = [title for title in book_df['title'] if query.lower() in title.lower()]
    # Sort the filtered titles alphabetically
    sorted_titles = sorted(filtered_titles)
    return sorted_titles

def get_song_details(track_id):
    token = get_token()
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    url = f'https://api.spotify.com/v1/tracks/{track_id}'
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        track_data = response.json()
        song_details = {
            'name': track_data['name'],
            'artist': ', '.join(artist['name'] for artist in track_data['artists']), 
            'album_name': track_data['album']['name'],
            'preview_url': track_data['preview_url'], 
            'external_urls': track_data['external_urls']['spotify']  
        }
        return song_details
    else:
        return f"Failed to retrieve song details: {response.status_code}"

# Helper function to return a formatted description of a book
def return_desc(query):
    filtered_df = book_df[(book_df['title'] == query)]
    #tokenize title
    if filtered_df.empty:
        print("Book does not exist in our database", query)
        return ""
    title_str = filtered_df['title'].iloc[0]
    desc_str = filtered_df['desc'].iloc[0]
    res = title_str + " " + desc_str
    return res

# Calculate cosine similarity, handling edge cases such as division by zero
def cossim(a, b):
    if np.isnan(norm(b)) or np.isnan(norm(a)) or norm(a) == 0 or norm(b) ==0 or not a or not b:
        return 0
    ans = dot(a, b) / (norm(a) * norm(b))
    return ans

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=8000)

