# Необходимые библиотеки
import joblib
import pandas as pd
from pathlib import Path
## Внутренние файлы
from preprocess import preprocess_df
from classification.vectorization import vectorize
from classification.make_preds import make_preds


# Переменные 
name_model = 'email_handler_module\\classification\\subm_embs_title_payload_lemmas_minilm.pkl'

# Загрузка модели
model = joblib.load(name_model)

if __name__ == "__main__":

    # Загрузка файла
    file_request = input("Введите путь/название файла в формате xlsx: ")
    df_request = pd.read_excel(file_request)
    
    df_clean = preprocess_df(df_request)

    df_embs = vectorize(df_clean)
    
    embs = df_embs.drop(columns=['id'])

    df_result = make_preds(model, embs, df_request)
    # Выгрузка файла
    name_file = Path(file_request).stem
    result_path_file = file_request.replace(name_file, f'{name_file}Result')
    df_result.to_excel(result_path_file)
    print(f'Работа выполнена, ваш файл: {result_path_file}')



