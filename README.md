# auto-avia-offer
Классификация сообщений по типу заявка не заявка.
<br>Выделение сущностей из письма.
<br>Название модели: subm_embs_title_payload_lemmas_minilm.pkl

В папке classification - скрипты для классификации заявка или не заявка.
<br> В extracting_entities ноутбук по извлечению сущностей.
<br> preprocess.py общий файл для предобработки
<br> subm_ensemble_minilm_title_payload_lemmas.csv файл с классификацией

### Инструкция по развёртыванию
Нужно создать виртуальное окружение, далее пример для windows:
<br/>virtualenv — утилита для создания виртуальных окружений.
<br/>Для установки виртуального окружения:
<br/>`pip install virtualenv`
<br/>Для создания виртуального окружения:
<br/>`virtualenv <имя виртуального окружения>`
<br/>В текущем каталоге будет создана новая директория с указанным вами названием, куда будут перенесены python, pip и в дальнейшем установлены другие библиотеки.
<br/>Активация виртуального окружения:
<br/>`<имя виртуального окружения>\Scripts\activate`
<br/>Для деактивации виртуального окружения:
<br/>`deactivate`

Нужно установить все необходимые библиотеки командой: `pip install -r email_handler_module\\requirements.txt`, также нужно скачать дополнительно модель для spacy `python -m spacy download ru_core_news_lg`.

Запустить модель в обвязке можно при помощи команды: `python email_handler_module\\main.py`. Когда она попросить ввести название файла, есть тестовый файл для проверки, его можно написать вот так `traind_data.xlsx`