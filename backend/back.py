from flask import Flask, request, jsonify, render_template
import random
from multiprocessing import Process, Manager, set_start_method, freeze_support
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variable to store the shared dictionary 'video_results'
video_results = None

# Allowed file extensions for video uploads
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def is_file_allowed(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/video', methods=['GET', 'POST'])
def upload_video():
    global video_results
    if video_results is None:
        return 'Server not ready', 500
    
    if request.method == 'GET':
        # Return HTML form for uploading a file
        return render_template('upload.html')
    
    if request.method == 'POST':
        # Check if a file is included in the request
        if 'file' not in request.files:
            return 'No file part', 400
        
        uploaded_file = request.files['file']
        
        # Check if a file was selected
        if uploaded_file.filename == '':
            return 'No selected file', 400
        
        # Validate the file type
        if not is_file_allowed(uploaded_file.filename):
            return 'Unsupported file type', 400
        
        # Generate a unique ID for the video
        video_id = random.randint(1, 1000000)
        # Secure the filename
        safe_filename = secure_filename(uploaded_file.filename)
        
        # Add a new entry to the results dictionary
        video_results[video_id] = [safe_filename, [], False]
        
        # Save the uploaded file temporarily
        if not os.path.exists('temp'):
            os.makedirs('temp')
        temp_file_path = f'temp/{safe_filename}'
        uploaded_file.save(temp_file_path)

        # Create and start a process to handle video processing
        process = Process(target=process_video, args=(video_id, safe_filename, video_results))
        process.start()

        return jsonify({'id': video_id}), 201

@app.route('/results/<int:video_id>', methods=['POST'])
def get_video_results(video_id):
    global video_results
    if video_results is None:
        return 'Server not ready', 500

    # For debugging: print the results of the video processing
    print(video_results[video_id][1])

    # Return the results of the video processing
    return jsonify({'results': video_results[video_id][1]}), 201

@app.route('/is_processing/<int:video_id>', methods=['GET'])
def check_processing_status(video_id):
    global video_results
    if video_results is None:
        return 'Server not ready', 500
    if video_id not in video_results:
        return jsonify({'error': 'Invalid ID'}), 404
    
    processing_status = video_results[video_id][2]  # Get processing status
    return jsonify({'processing': processing_status}), 201  # Return status as JSON

def determine_tag_levels(tags):
    """Determine the levels of tags based on a CSV file."""
    tag_levels = []
    try:
        df = pd.read_csv('taggi.csv')
    except FileNotFoundError:
        print("File 'taggi.csv' not found.")
        return tag_levels  # Return an empty list if the file is not found

    for tag in tags:
        for row in df.values:
            try:
                tag_levels.append([tag, list(row).index(tag) + 1])
                break
            except ValueError:
                pass
    return tag_levels

def process_video(video_id, filename, video_results):
    """Process the uploaded video in a separate process."""
    try:
        # Example of tags for processing
        tags = ['Автомобили класса люкс', 'Карьера', 'Домашние задания', 'Головоломки', 'Конный спорт', 'Спорт', 'Автогонки', 'Гребля', 'Регби', 'Красота', 'Аксессуары', 'Языки программирования', 'Мультфильмы и анимация', 'Реалити-ТВ', 'Мобильные игры', 'Тип путешествия', 'Игры']
        
        # Determine levels for the tags
        tag_levels = determine_tag_levels(tags)
        
        # Update the global results dictionary
        video_results[video_id] = [filename, tag_levels, True]
        
        # Remove the temporary file after processing
        temp_file_path = f'temp/{filename}'
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception as e:
        print(f'Error processing video {video_id}: {e}')
        video_results[video_id][2] = False  # Mark processing as failed

def main():
    global video_results
    # Set the start method to 'spawn' for compatibility with Windows
    set_start_method('spawn')
    # Support freezing (for Windows)
    freeze_support()
    # Initialize the Manager and shared dictionary 'video_results'
    manager = Manager()
    video_results = manager.dict()
    # Run the Flask application without debug mode and reloader
    app.run(debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
