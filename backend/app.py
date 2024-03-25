import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
# from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script, 
#assumming data is stored in 'GoodReads_condensed.json' and 'Spotify_condensed.json'

goodreads_file_path = os.path.join(current_directory, 'GoodReads_condensed.json')

spotify_file_path = os.path.join(current_directory, 'Spotify_condensed.json')


with open(goodreads_file_path, 'r') as file:
    book_data = json.load(file)
    book_df = pd.DataFrame(book_data)

with open(spotify_file_path, 'r') as file:
    spotify_data = json.load(file)
    spotify_df = pd.DataFrame(spotify_data)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

#Manual mapping of book genres to song genres (First prototype only)
book_to_song_genres = {'Nonfiction': ['folk', ' indie'], 'Fiction': [' indie', ' pop', 'pop'],
'Fantasy': [' pop', 'electronic', 'pop', ' indie'], 'Romance': [' pop', ' 90s'], 'Historical': [' classic rock', ' blues'],
'History': [' classic rock', ' blues'], 'Childrens': [' pop',  'pop'], 'Cultural': [' 90s', ' 70s'], 
'Sequential Art': ['electronic', 'rock', ' indie'], 'Mystery': ['jazz'], 'Religion': [' indie', 'folk'], 
'Science': ['electronic', 'rock'], 'Science Fiction': [' pop', ' indie rock', 'electronic'], 
'Paranormal': [' classic rock', 'gothic metal', 'hard rock']}

#Get most popular song genres and most popular book genres in our datasets (helping to map genres)
def popular_genres():
    #Unionize the songs
    song_genres_count = {}
    for genre_list in spotify_df['tagstokenized']:
        for genre in genre_list:
            old_count = song_genres_count.get(genre, 0)
            song_genres_count[genre] = old_count+1
    book_genres_count = {}
    for genre_list in book_df['genretokenized']:
        for genre in genre_list:
            old_count = book_genres_count.get(genre, 0)
            book_genres_count[genre] = old_count+1
    sorted_song_genre_count = (sorted(song_genres_count.items(), key=lambda x: x[1], reverse= True))
    sorted_book_genres_count = (sorted(book_genres_count.items(), key=lambda x: x[1], reverse= True))
    for pair in sorted_book_genres_count[:20]:
            print(pair)
    for pair in sorted_song_genre_count[:20]:
            print(pair)
    
#Filter songs by genre (if filtered is empty, just return all songs)

#Tokenize the title and description of a book based on query title given

def tokenize_book(query):
    filtered_df = book_df[(book_df['title'] == query)]
    #tokenize title
    if filtered_df.empty:
        print("Book does not exist in our database", query)
        return []
    title_str = filtered_df['title'].iloc[0]
    title_tok = re.findall(r'\w+\'?\w*', title_str)
    desc_str = filtered_df['desc'].iloc[0]
    desc_tok = re.findall(r'\w+\'?\w*', desc_str)
    res = title_tok + desc_tok
    return res

#Tokenize the title and text of a song by row id given
def tokenize_song_by_i(i):
    title_str = spotify_df['title'].iloc[i]
    title_tok = re.findall(r'\w+\'?\w*', title_str)
    text_str = spotify_df['text'].iloc[i]
    text_tok = re.findall(r'\w+\'?\w*', text_str)
    res = title_tok + text_tok
    return res

#Filtering spotify data based on book genre
#query = book of interest
def song_filter(query):
    book = book_df[(book_df['title'] == query)]
    our_genres = book['genretokenized']
    rel_song_genre_set = set()
    for key, s_gen in book_to_song_genres.items():
        for genre in our_genres:
            if key == genre:
                rel_song_genre_set.union(set(s_gen))
    if (len(rel_song_genre_set)==0):
        return spotify_df
    rel_song_genre_list = list(rel_song_genre_set)
    filtered_df = spotify_df[spotify_df['tagstokenized'].apply(lambda x: all(query in x for query in rel_song_genre_list))]
    return filtered_df

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

#Route to return top ten similar songs to a book (based on cossine similarity)
@app.route("/songs", methods = ['GET'])
def episodes_search():
    #popular_genres()
    text = request.args.get("title")
    #Tokenize title and desc of requested book 
    tokenize_req_book = tokenize_book(text)
    book_vector = build_vector(tokenize_req_book, tokenize_req_book)
  #  print(book_vector)
    #Initialize ranking dictionary of cossine similarities
    cos_sim_ranking = {}
    song_count = spotify_df.shape[0]
    print(song_count)
    #Have to get the cossine sim between each song and book, put in map, and then sort the map
    for i in range(song_count):
        #Get tokenized title and text for song i
        tokenized_song = tokenize_song_by_i(i)
        #Convert our song to a vector to call cossim on 
        song_vector = build_vector(tokenized_song, tokenize_req_book)
       # print(song_vector)
        cossim_measure = cossim(book_vector, song_vector)
       # print(cossim_measure)
        cos_sim_ranking[i] = cossim_measure

    #Sort our cos_sim_ranking by cos_sim (greatest to least)
    sorted_cos_sim = (sorted(cos_sim_ranking.items(), key=lambda x: x[1], reverse= True))
    response_json = {}
    response_json["top_ten_songs"] = []
    for i in range(10):
        #Get title
        id = sorted_cos_sim[i][0]
        song = spotify_df.iloc[id].title
        #Add to response_json
        response_json["top_ten_songs"].append(song)
        print(song)
    return json.dumps(response_json), 200

#Builds Frequency Vector 

def build_vector(doc, query_words):
    
    frequency_vector= [0] * len(query_words)
    
    # Create a dictionary to store the index of each word in the query vocabulary
    word_index = {word: i for i, word in enumerate(query_words)}

    # Update the counts in the frequency vector for words present in both doc and query_words
    for word in doc:
        if word in word_index:
            index = word_index[word]
            frequency_vector[index] += 1

    return frequency_vector

def cossim(a, b):
  #  print(a,b)
    if np.isnan(norm(b)) or np.isnan(norm(a)) or norm(a) == 0 or norm(b) ==0 or not a or not b:
        return 0
    ans = dot(a, b) / (norm(a) * norm(b))
    return ans

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=8000)