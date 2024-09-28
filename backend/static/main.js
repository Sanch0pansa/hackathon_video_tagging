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

let files = [];
function handleFiles() {

    const fileList = this.files;
    Array.from(fileList).forEach(element => {
        files.push(element);
        let div = document.createElement("div");
        let button = document.createElement("button");
        
        let tag = document.getElementById("videos");
        let text = document.createTextNode(element.name);
        button.dataset.file = files.length - 1;
        button.addEventListener('click', function(){
            
            files.splice(this.dataset.file, 1);
            button.parentNode.remove();
        })
        div.classList.add('container-videos');
        button.classList.add('remove-video');
        button.type="button";
        div.appendChild(text);
        div.appendChild(button);
        tag.appendChild(div);
    });   
}

function handleSubmit(event){
    event.preventDefault();
    const data = new FormData();

    for(const file of files){
        data.append('files[]', file, file.name);
    }

    return fetch('/', {
        method: 'POST',
        body: data,
    });

    
}

myForm.addEventListener('submit', handleSubmit);

