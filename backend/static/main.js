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
            let stage_1 = document.querySelector('#stage-1');
            let stage_2 = document.querySelector('#stage-2');
            stage_1.style.display = 'none';
            stage_2.style.display = 'block';
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
            let ready = response.json();
            if(ready.processing === false){
                console.log('Успех');
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

