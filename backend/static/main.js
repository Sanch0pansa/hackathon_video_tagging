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

function handleSubmit(event){
    event.preventDefault();
    const data = new FormData();


    data.append('file', file_1, file_1.name);

    return fetch('/video', {
        method: 'POST',
        body: data,
    });

    
}

myForm.addEventListener('submit', handleSubmit);

