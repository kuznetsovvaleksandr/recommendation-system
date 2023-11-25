import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv


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


print('Введите кол-во книг, которые хотите получить')
k = int(input())
print('Введите тему, которая вам интересна')
my_theme = input()

rec_list = rec_sys(my_theme, k)
col_list = ['COS_SIM', 'BOOK']
rec_df = pd.DataFrame(data=rec_list, columns=col_list)

print(rec_df)



