// получение кнопки "Обработать" и формы
let button = document.body.querySelector(".handler");
const myForm = document.getElementById('form-video');

// получение элемента input для загрузки файлов
const fileInput = document.getElementById('add-file');

// Обработчик события для изменения состояния кнопки 
button.addEventListener('mousedown', function() {
    button.classList.add('active'); // Добавляем класс active
});

// Обработчик события для удалени класса при отпускании кнопки
button.addEventListener('mouseup', function() {
    button.classList.remove('active');
});

// Обработчик события для удалени класса  при уходе мыши с кнопки
button.addEventListener('mouseleave', function() {
    button.classList.remove('active');
});

// Обработчик события изменения в input для файлов
fileInput.addEventListener("change", handleFiles);

// Переменная для хранения загруженного файла
let file_1;

// Функция обработки выбранного файла
function handleFiles() {
    file_1 = this.files[0];
    let div = document.createElement("div");
    let button = document.createElement("button");
    let tag = document.getElementById("videos");
    let text = document.createTextNode(file_1.name);

    // Обработчик для кнопки удаления
    button.addEventListener('click', function() {
        fileInput.value = '';
        file_1 = "";
        button.parentNode.remove();
        console.log(file_1);
    });

    div.classList.add('container-videos');
    button.classList.add('remove-video');
    button.type = "button";
    div.appendChild(text);
    div.appendChild(button);
    tag.appendChild(div);
    console.log(file_1);
}

// Этапы обработки видео
let stage_1 = document.querySelector('#stage-1');
let stage_2 = document.querySelector('#stage-2');
let stage_3 = document.querySelector('#stage-3');

// Функция обработки отправки формы
async function handleSubmit(event) {
    event.preventDefault();
    const data = new FormData();
    data.append('file', file_1, file_1.name);

    try {
        let response = await fetch('/video', {
            method: 'POST',
            body: data,
        });

        if (response.ok) {
            let result = await response.json();
            console.log(result);
            stage_1.style.display = 'none';
            stage_2.style.display = 'block';
            change_text();
            processing(result.id);
        }
    } catch (error) {
        console.error('Ошибка при отправке данных на сервер:', error);
    }    
}

// Функция проверки состояния обработки видео
async function processing(id_video) {
    try {
        let response = await fetch('/is_processing/' + id_video, { method: 'GET' });

        if (response.ok) {
            let ready = await response.json();
            console.log(ready);
            if (ready.processing === true) {
                stage_2.style.display = 'none';
                stage_3.style.display = 'block';
                add_tags(id_video);
            } else {
                setTimeout(() => processing(id_video), 1000);
            }
        }
    } catch (error) {
        console.error('Ошибка при отправке данных на сервер:', error);
    }    
}

myForm.addEventListener('submit', handleSubmit);
let tags = document.querySelector('.tags');

// Функция для добавления тегов после обработки видео
async function add_tags(id_video) {
    try {
        let response = await fetch('/results/' + id_video, { method: 'POST' });

        if (response.ok) {
            let data = await response.json();
            console.log(data);
            data = data.res;
            data.forEach(element => {
                let tag = document.createElement('div');
                tag.classList.add('tag', 'tag-' + element[1]);
                tag.append("#" + element[0]);
                tags.appendChild(tag);
            });
        }
    } catch (error) {
        console.error('Ошибка при получении тегов:', error);
    }
}

let text_stage_2 = [
    "Подождите, видео обрабатывается",
    "Пожалуйста, подождите",
    "Подождите одну минуту",
    "Ладно, не одну",
    "Подождите ещё чуть-чуть",
    "Ваше видео точно будет обработано",
    "Мы почти закончили",
    "Осталось ещё немного"
];

let cur_char = 0;
let h1_stage_2 = document.getElementById("text-load");
let number_cur_text = 0;
let direction_forward = true;

// Функция для изменения текста
function change_text() {
    if (direction_forward) {
        h1_stage_2.innerHTML += text_stage_2[number_cur_text][cur_char];
        cur_char++;
        if (cur_char == text_stage_2[number_cur_text].length) {
            direction_forward = false;
            setTimeout(change_text, 2000);
        } else {
            setTimeout(change_text, 100);
        }
    } else {
        h1_stage_2.innerHTML = h1_stage_2.innerHTML.slice(0, -2);
        cur_char--;
        if (cur_char == 0) {
            direction_forward = true;
            number_cur_text++;
            number_cur_text %= text_stage_2.length;
            setTimeout(change_text, 100);
        } else {
            setTimeout(change_text, 100);
        }
    }
}

