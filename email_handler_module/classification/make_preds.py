'''
Модуль для предсказания заявка сообщение или нет.
=================================================

Функции:
--------
make_preds(model, embs, df, threshold=0.5, mark_labels=False) `->` df_result
get_labeL_not_request(df_result) `->` df_result

Зависимости:
------------
numpy

Дата:
-----
25.05.2023

Авторы:
-------
Черников Дмитрий
\n Шилова Надежда

'''


# Необходимые библиотеки
import numpy as np


def get_labeL_not_request(df_result):
    '''Функция отметки автоматически генерируемых сообщений
    
    Параметры:
    ----------
    df_result : DataFrame
        датафрейм с исходными сообщениями
    
    Возращаемые значения:
    ---------------------
    df_result : DataFrame
        датафрейм с заполненным столбцом 0\1 в label

    '''
    df_result.loc[
        df_result['title'].str.contains(
            '^Электронный билет:', na=False,
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^Ваучер к заказу', na=False,)
        ) & (
            df_result['text'].str.contains('^Уважаемый \w+! Благодарим Вас за обращение в компанию Аэроклуб', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        df_result['title'].str.contains(
            '^Подтверждение бронирования №', na=False,
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^i’way для', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^Оповещение ASIM по заказу', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^Поставщик внес изменения в стоимость брон', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        df_result['title'].str.contains(
            '^Please purchase tickets according to the booking', na=False,
        ), 'label'
    ] = 1

    df_result.loc[
        (
            df_result['title'].str.contains('^Voucher for order', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        df_result['title'].str.contains(
            '^Прошу оформить билеты по бронированию', na=False,
        ), 'label'
    ] = 1

    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub.*Сообщение от Аэроклуб АО', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub', na=False,)
        ) & (
            df_result['text'].str.contains('оформил услугу', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub', na=False,)
        ) & (
            df_result['text'].str.contains('пытался забронировать услугу', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub', na=False,)
        ) & (
            df_result['text'].str.contains('просит оформить услугу', na=False,)
        ), 'label'
    ] = 1

    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub', na=False,)
        ) & (
            df_result['text'].str.contains('пытался произвести изменение услуги', na=False,)
        ), 'label'
    ] = 0

    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub', na=False,)
        ) & (
            df_result['text'].str.contains('пытался произвести отмена услуги', na=False,)
        ), 'label'
    ] = 0
    
    df_result.loc[
        (
            df_result['title'].str.contains('^\[ Aeroclub', na=False,)
        ) & (
            df_result['text'].str.contains('пытался произвести бронирование услуги', na=False,)
        ), 'label'
    ] = 0
    
    df_result.loc[
        df_result['text'].str.contains(
            '^Здравствуйте, aeroclubXML', na=False,
        ), 'label'
    ] = 0
    return df_result


def make_preds(model, embs, df, threshold=0.5, mark_labels=False):
    '''Функция для предсказания заявка сообщение или нет.
    =====================================================

    Параметры:
    ----------
    model :
        обученная модель классификации
    embs : DataFrame
        с почитанными эмбенденгами
    df : DataFrame
        с таблицей сообщений
    threshold : float, по умолчанию 0.5
        используется для преобразования вероятностей proba в бинарные значения
    mark_labels : bool, по умолчанию False
        уточнение предсказания по отбивкам после модели
    
    Возвращаемые значения:
    ----------------------
    df : DataFrame
        с предсказанием 0/1 label заявка или не заявка сообщение

    '''

    proba = model.predict_proba(embs)[:,1]
    preds = np.where(proba > threshold, 1, 0)

    df_result = df.copy()
    df_result[f'label'] = preds

    if mark_labels:
        df_result = get_labeL_not_request(df_result)
    
    return df_result
