import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
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
CORS(app)


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

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

#Route to return top ten similar songs to a book (based on cossine similarity)
@app.route("/songs", methods = ['GET'])
def episodes_search():
    text = request.args.get("title")
    #Tokenize title and desc of requested book 
    tokenize_req_book = tokenize_book(text)
    book_vector = build_vector(tokenize_req_book, tokenize_req_book)
    #Initialize ranking dictionary of cossine similarities
    cos_sim_ranking = {}
    song_count = len(spotify_df['title'])
    #Have to get the cossine sim between each song and book, put in map, and then sort the map
    for i in range(song_count):
        #Get tokenized title and text for song i
        tokenized_song = tokenize_song_by_i(i)
        #Convert our song to a vector to call cossim on 
        song_vector = build_vector(tokenized_song, tokenize_req_book)
        cossim_measure = cossim(book_vector, song_vector)
        cos_sim_ranking[i] = cossim_measure

    #Sort our cos_sim_ranking by cos_sim (greatest to least)
    sorted_cos_sim = (sorted(cos_sim_ranking.items(), key=lambda x: x[1]))
    response_json = {}
    response_json["top_ten_songs"] = []
    for i in range(10):
        #Get title
        id = sorted_cos_sim[i][0]
        title = spotify_df['title'].iloc[id]  
        #Add to response_json
        response_json["top_ten_songs"].append(title)

    return response_json

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
    ans = dot(a, b) / (norm(a) * norm(b))
    if not ans or np.isnan(ans):
        return 0
    return ans

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=6000)
