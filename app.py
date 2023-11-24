import streamlit as st
import pandas as pd
import numpy as np
import csv
from sentence_transformers import SentenceTransformer, util

# Модель
#___________________________________________________________________________________
def read_df():
    df = pd.read_csv('data.csv')
    pd.options.mode.chained_assignment = None
    return df

df = read_df()

sentence = SentenceTransformer('all-MiniLM-L6-v2')

def read_csv(filename):
    with open(filename, newline='') as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

embeddings = read_csv('embeddings.csv')

def rec_sys(book_theme, n):

    df['combined_features'].iloc[-1], df['Book'].iloc[-1] = book_theme, book_theme

    df.reset_index(drop=True, inplace=True)

    def emb(embed):
        embedding = list(embed)
        embedding_theme = sentence.encode(df['combined_features'].iloc[-1])
        embedding_theme = list(embedding_theme)
        embedding.append(embedding_theme)
        embedding = np.array(embedding)
        return embedding

    embedding = emb(embeddings)
    cos_sim = util.cos_sim(embedding, embedding)

    def get_recommendations(book_id, similarity_matrix):
        similar_books = list(enumerate(similarity_matrix[book_id]))
        similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
        similar_books = similar_books[1:n + 1]
        recommended_books = [df.iloc[i[0]]["Book"] for i in similar_books]
        return recommended_books

    book_id = df[df['Book'] == book_theme].index[0]
    similar = list(enumerate(cos_sim[book_id]))
    similar = sorted(similar, key=lambda x: x[1], reverse=True)
    similar = similar[0:n + 1]
    recommended_books = get_recommendations(book_id, cos_sim)

    rec_list = list([] for i in range(n))

    print('Тема для поиска', book_theme)
    for i in range(len(recommended_books)):
        rec_list[i].append(float(similar[i + 1][1]))
        rec_list[i].append(recommended_books[i])

    return rec_list
#___________________________________________________________________________________



#Веб интерфейс
#___________________________________________________________________________________

st.title('BOOK REC')

books = st.text_input("Введите кол-во книг, которые хотите получить")
result = books.title()

if(st.button('Submit')):
    st.success(result)

theme = st.text_input("Введите интересующую тему (английский язык)")
theme_1 = theme.title()

if(st.button('GO!')):
    rec_list = rec_sys(theme_1, int(result))
    col_list = ['COS_SIM', 'BOOK']
    rec_df = pd.DataFrame(data=rec_list, columns=col_list)
    st.write(rec_df)

