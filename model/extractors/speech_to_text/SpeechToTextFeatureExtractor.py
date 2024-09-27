import torch
from transformers import AutoTokenizer, AutoModel
from moviepy.editor import VideoFileClip
from vosk import Model, KaldiRecognizer
import json
import os
import wave


class SpeechToTextFeatureExtractor:
    def __init__(self, embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", vosk_model_path="path/to/vosk/russian/model"):
        # Embedding model (changed to multilingual model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name, cache_dir="./cache")
        self.model = AutoModel.from_pretrained(
            embedding_model_name, cache_dir="./cache")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        # Vosk setup (use Russian model)
        self.vosk_model = Model(vosk_model_path)

    def extract_audio(self, video_path):
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        return audio_path

    def audio_to_text(self, audio_path):
        # Convert audio to text using Vosk
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(self.vosk_model, wf.getframerate())

        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text += result.get("text", "") + " "

        final_result = json.loads(rec.FinalResult())
        text += final_result.get("text", "")
        return text.strip()

    def get_embeddings(self, text):
        # Get embeddings from text
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def extract_features(self, video_path):
        # Extract features from video
        audio_path = self.extract_audio(video_path)
        text = self.audio_to_text(audio_path)
        embeddings = self.get_embeddings(text)

        # Clean up temporary audio file
        os.remove(audio_path)

        return embeddings
