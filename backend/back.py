from flask import Flask, request, jsonify, render_template
import sqlalchemy as db
import random
from multiprocessing import Process, Manager, Event
import time
import pandas as pd

app = Flask(__name__)


res = {136584:['name', [['dfghj', 1], ['rtyui', 2]], False]}
@app.route('/video', methods=['GET', 'POST'])
def post_vid():
    if request.method == 'GET':
        # Возвращаем HTML-форму
        return render_template('upload.html')
    if request.method == 'POST':
        # Проверяем, есть ли файл в запросе
        if 'file' not in request.files:
            #!!!!!!!!!!!!!!!!!!!!!HTML
            return 'No file part', 400
        file = request.files['file']
        # Если файл не выбран
        if file.filename == '':
            return 'No selected file', 400

@app.route('/results', methods=['POST'])
def res_vid():
    file = request.files['file']
    flagg = True
    for key in res:
        if res[key][0] == file.filename:
            flagg = False
            break
    if flagg:
        # Сохраним файл временно
        file_path = f'temp/{file.filename}'  # Путь для сохранения файла
        file.save(file_path)  # Сохранение файла

        id = random.randint(1, 1000000) #генерим айди
        while (id in res):
            id = random.randint(1, 1000000)
        res[id] = ['', '', False]
        #ПОТОК ОБРАБОТКИ ВИДЕО
        manager = Manager()  # для работы со списком
        tegs = manager.list() #теги на выходе
        finish_li = Event()
        p = Process(target=f, args=(file_path, tegs, finish_li))
        p.start()
        #finish_li.is_set()#Щас false
        p.join()
        #finish_li.is_set()#Щас true

        name = file.filename #имя видоса

        #Определение уровня тега
        level = level_teg(tegs) #на выход список списков
        res[id] = [name, level, True]

        #есть список тегов с видоса
        print(id)
        print(res)
        return jsonify(({'tegs' : res})), 201
    return jsonify({'error': 'File already exists'}), 400


#УРОВНИ ТЕГОВ
def level_teg(tegi):
    lev_teg = []
    df = pd.read_csv('taggi.csv')
    for a in tegi:
        for i in df.values:
            try:
                lev_teg.append([a, list(i).index(a) + 1])
                break
            except ValueError:
                pass
    return lev_teg


def f(file_path, tegs, finish_li):
    #лютая обработка видоса
    #на выход теги!!
    time.sleep(2)#тип обработка
    tags = ['Автомобили класса люкс', 'Карьера', 'Домашние задания', 'Головоломки']
    tegs.extend(tags)
    finish_li.set()


@app.route('/is_processing/<int:id>', methods=['GET'])
def get_status(id):
    if not res[id]:
        return jsonify({'error': 'Invalid ID'}), 404
    res[id][2] = True
    processing_status = res[id][2]  # Получаем статус обработки
    return jsonify({'processing': processing_status}), 200  # Возвращаем статус в виде JSON

if __name__ == '__main__':
    app.run(debug=True)