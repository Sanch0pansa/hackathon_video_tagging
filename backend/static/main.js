let button = document.body.querySelector(".handler");
const myForm = document.getElementById('form-video');

const fileInput = document.getElementById('add-file');


button.addEventListener('mousedown', function(){
    button.classList.add('active');
});

button.addEventListener('mouseup', function(){
    button.classList.remove('active');

});

button.addEventListener('mouseleave', function() {
    button.classList.remove('active');
});

fileInput.addEventListener("change", handleFiles);

let file_1;
function handleFiles() {

    file_1 = this.files[0];
    let div = document.createElement("div");
    let button = document.createElement("button");
    let tag = document.getElementById("videos");
    let text = document.createTextNode(file_1.name);
    
    button.addEventListener('click', function(){
        fileInput.value = '';
        file_1="";
        button.parentNode.remove();
        console.log(file_1);
    })

    div.classList.add('container-videos');
    button.classList.add('remove-video');
    button.type="button";
    div.appendChild(text);
    div.appendChild(button);
    tag.appendChild(div);
    console.log(file_1);
       
}

let stage_1 = document.querySelector('#stage-1');
let stage_2 = document.querySelector('#stage-2');
let stage_3 = document.querySelector('#stage-3');

async function handleSubmit(event){
    event.preventDefault();
    const data = new FormData();


    data.append('file', file_1, file_1.name);

    try{

        let response = await fetch('/video', {
            method: 'POST',
            body: data,
        });

        if(response.ok){
            let result = await response.json();
            console.log(result);
            
            stage_1.style.display = 'none';
            stage_2.style.display = 'block';
            change_text();
            processing(result.id);
        }
    }
    catch (error) {
        console.error('Ошибка при отправке данных на сервер:', error);
    }    
} 

async function processing(id_video){
    try{

        let response = await fetch('/is_processing/' + id_video, {
            method: 'GET',
        });

        if(response.ok){
            let ready = await response.json();
            console.log(ready);
            if(ready.processing === true){
                stage_2.style.display = 'none';
                stage_3.style.display = 'block';
                add_tags(id_video);
            }
            else{
                setTimeout(() => processing(id_video), 1000)
            }
        }
    }
    catch (error) {
        console.error('Ошибка при отправке данных на сервер:', error);
    }    
}
myForm.addEventListener('submit', handleSubmit);

let tags = document.querySelector('.tags');
async function add_tags(id_video){
    try{
        let response = await fetch('/results/' + id_video, {
            method: 'POST',
        });
        if(response.ok){
            let data = await response.json();

    
            console.log(data);
            
            data = data.res;
            data.forEach(element => {
                console.log(element);
                if(element[1] == 1){
                    let tag_1 = document.createElement('div');
                    tag_1.classList.add('tag');
                    tag_1.classList.add('tag-1');
                    tag_1.append("#" +element[0]);
                    tags.appendChild(tag_1);
                }
                else if(element[1] == 2){
                    let tag_2 = document.createElement('div');
                    tag_2.classList.add('tag');
                    tag_2.classList.add('tag-2');
                    tag_2.append("#" +element[0]);
                    tags.appendChild(tag_2);

                }
                else{
                    let tag_3 = document.createElement('div');
                    tag_3.classList.add('tag');
                    tag_3.classList.add('tag-3');
                    tag_3.append("#"+element[0]);
                    tags.appendChild(tag_3);
                }
            });

        }
    }
    catch{

    }
}

let text_stage_2 = ["Подождите, видео обрабатывается",
    "Пожалуйста, подождите",
    "Подождите одну минуту",
    "Ладно, не одну",
    "Подождите ещё чуть-чуть",
    "Ваше видео точно будет обработано",
    "Мы почти закончили",
    "Осталось ещё немного"];

let cur_char = 0;
let h1_stage_2 = document.getElementById("text-load");
let number_cur_text =0;
let direction_forward = true; 

function change_text() {
    if(direction_forward){
        h1_stage_2.innerHTML += text_stage_2[number_cur_text][cur_char];
        cur_char++;
        if(cur_char== text_stage_2[number_cur_text].length){
            direction_forward = false;
            setTimeout(change_text, 2000);
        }
        else{
            setTimeout(change_text, 100);
        }
    }
    else{
        h1_stage_2.innerHTML = h1_stage_2.innerHTML.slice(0, h1_stage_2.innerHTML.length-2);
        cur_char--;
        if(cur_char==0){
            direction_forward = true;
            number_cur_text++;
            number_cur_text %= text_stage_2.length;
            setTimeout(change_text, 100);
        }
        else{
            setTimeout(change_text, 100);
        }
    }
}


