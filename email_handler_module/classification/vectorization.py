'''
Модуль для подготовки к модели классификации.
=============================================

Функции:
--------
vectorize(df) `->` df

Зависимости:
------------
numpy
\npandas
\nsentence_transformers

Дата:
-----
25.05.2023

Авторы:
-------
Черников Дмитрий

'''


# Необходимые библиотеки
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def vectorize(df):
    '''Функция считает эмбенденги.
    ==============================

    Параметры:
    ----------
    df : DataFrame
        после препроцессинга

    Возвращаемые значения:
    ----------------------
    df : DataFrame
        с почитанными эмбенденгами

    '''

    vectorizer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    docs = df['title'].fillna('')
    title_embs = vectorizer.encode(docs, device='cpu', show_progress_bar=True)

    docs = df['payload'].fillna('')
    text_embs = vectorizer.encode(docs, device='cpu', show_progress_bar=True)

    embs = np.hstack((title_embs, text_embs))

    df_embs = pd.DataFrame(embs, index=df.index).reset_index().rename(columns={'index':'id'})
    df_embs = df_embs.merge(df[['id']], how='inner', on='id')

    return df_embs
