from flask import Flask, request, jsonify, render_template
import random
from multiprocessing import Process, Manager, set_start_method, freeze_support
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Глобальная переменная для общего словаря res
res = None

# Дополнительно: определение допустимых расширений файлов
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    """Проверяет, имеет ли файл допустимое расширение."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/video', methods=['GET', 'POST'])
def post_vid():
    global res
    if res is None:
        return 'Server not ready', 500
    if request.method == 'GET':
        # Возвращаем HTML-форму для загрузки файла
        return render_template('upload.html')
    if request.method == 'POST':
        # Проверяем наличие файла в запросе
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        # Проверяем, выбран ли файл
        if file.filename == '':
            return 'No selected file', 400
        # Проверяем допустимость файла
        if not allowed_file(file.filename):
            return 'Unsupported file type', 400
        # Генерируем уникальный ID
        id = random.randint(1, 1000000)
        # Безопасное имя файла
        name = secure_filename(file.filename)
        # Добавляем запись в res
        res[id] = [name, [], False]
        # Сохраняем файл временно
        if not os.path.exists('temp'):
            os.makedirs('temp')
        file_path = f'temp/{name}'
        file.save(file_path)

        # Создаем и запускаем процесс обработки видео
        p = Process(target=process_video, args=(id, name, res))
        p.start()

        return jsonify({'id': id}), 201


@app.route('/results/<int:id>', methods=['POST'])
def res_vid(id):
    global res
    if res is None:
        return 'Server not ready', 500

    # Для отладки
    print(res[id][2])

    # Проверяем тип res[id][2]
    if isinstance(res[id][2], dict):
        return jsonify({'res': dict(res[id][2])}), 201
    elif isinstance(res[id][2], bool):
        return jsonify({'status': res[id][2]}), 200
    else:
        return jsonify({'error': 'Expected a dictionary or boolean but got a {}'.format(type(res[id][2]))}), 400

    # Если все условия не выполнены (хотя это маловероятно), добавим явный возврат
    return jsonify({'error': 'Unexpected error occurred'}), 500

@app.route('/is_processing/<int:id>', methods=['GET'])
def get_status(id):
    global res
    if res is None:
        return 'Server not ready', 500
    if id not in res:
        return jsonify({'error': 'Invalid ID'}), 404
    processing_status = res[id][2]  # Получаем статус обработки
    return jsonify({'processing': processing_status}), 201  # Возвращаем статус в виде JSON


# Функция для определения уровней тегов
def level_teg(tegi):
    lev_teg = []
    try:
        df = pd.read_csv('taggi.csv')
    except FileNotFoundError:
        print("Файл 'taggi.csv' не найден.")
        return lev_teg  # Возвращаем пустой список, если файл не найден

    for a in tegi:
        for i in df.values:
            try:
                lev_teg.append([a, list(i).index(a) + 1])
                break
            except ValueError:
                pass
    return lev_teg


# Функция обработки видео, выполняемая в отдельном процессе
def process_video(id, filename, res):
    try:
        # Тип обработка
        tags = ['Автомобили класса люкс', 'Карьера', 'Домашние задания', 'Головоломки']
        # Определяем уровни тегов
        lev_tags = level_teg(tags)
        # Обновляем общий словарь res
        res[id] = [filename, lev_tags, True]
        # удаление временного файла после обработки
        file_path = f'temp/{filename}'
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f'Ошибка при обработке видео {id}: {e}')
        res[id][2] = False  # Указываем, что обработка не удалась

def main():
    global res
    # Устанавливаем метод запуска 'spawn' для совместимости с Windows
    set_start_method('spawn')
    # Поддержка заморозки (для Windows)
    freeze_support()
    # Инициализируем Manager и общий словарь res
    manager = Manager()
    res = manager.dict()
    # Запускаем Flask-приложение без режима отладки и перезагрузчика
    app.run(debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
