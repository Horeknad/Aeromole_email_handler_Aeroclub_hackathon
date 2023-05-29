'''
Модуль для предпроцессинга предоставляемых данных.
==================================================

Функции:
--------
preprocess_body(text) `->` text
lemmatize_text(text) `->` text
preprocess_df(df) `->` df

Переменные:
-----------
disable
\nnlp
\ndf_dictionary_airport

Зависимости:
------------
pandas
\nspacy

Дата:
-----
25.05.2023

Авторы:
-------
Черников Дмитрий

'''


# Необходимые библиотеки
import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()

disable = ['tok2vec', 'tagger', 'morphologizer', 'parser', 'senter', 'attribute_ruler', 'ner']

nlp = spacy.load("ru_core_news_lg", disable=disable)

def preprocess_body(text):
    '''Функция для выделения текста сообщения.
    ==========================================

    Параметры:
    ----------
    text : str
        текст сообщения

    Возвращаемые значения:
    ----------------------
    text : str
        очищенное сообщение от цитирования и подписей

    '''
    text = text.lower()
    
    # убираю часть после re: когда цитируется сообщение, на которое составляется ответ
    text = text.split('re:', 1)[0]
    text = text.split(' re ', 1)[0]
    
    # убираю подписи
    text = text.split('с уважением', 1)[0]
    text = text.split('this e-mail and its attachments', 1)[0]
    text = text.split('this e-mail and all attachments', 1)[0]
    text = text.split('this message and its attachments', 1)[0]
    text = text.split('this message and all attachments', 1)[0]
    text = text.split('information contained in this email', 1)[0]
    
    return text


def lemmatize_text(text):
    '''Функция для лемматизации слов в сообщение'''
    doc = nlp(text)
    return ' '.join([t.lemma_ for t in doc])


def preprocess_df(df):
    '''Функция производит препроцессинг по всем текстовым столбцам таблицы.
    =======================================================================

    Параметры:
    ----------
    df : DataFrame
        содержит письма

    Возвращаемые значения:
    ----------------------
    df : DataFrame
        очищенное и лемматизированные сообщения, лемматизируемые заголовки

    '''
    title = df['title'].str.lower().fillna('')

    print('Extracting payload...')
    payload = df['text'].fillna('').progress_apply(preprocess_body).rename('payload')

    print('Lemmatizing text...')
    title_lemmas = title.progress_apply(lemmatize_text).rename('title_lemmas')
    payload_lemmas = payload.progress_apply(lemmatize_text).rename('payload_lemmas')
    
    title_payload = title.rename('title_payload')
    title_payload += ' payload '
    title_payload += payload

    title_payload_lemmas = title_lemmas.rename('title_payload_lemmas')
    title_payload_lemmas += ' payload '
    title_payload_lemmas += payload_lemmas

    result = pd.concat((df, payload, title_payload, title_lemmas, payload_lemmas, title_payload_lemmas), axis=1)

    return result