import torch
import cv2
from transformers import AutoImageProcessor, VideoMAEModel


class VideoFeatureExtractor:
    def __init__(self, *args, **kwargs):
        # Initialize the image processor and model
        self.image_processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base", cache_dir="./cache")
        self.model = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base", cache_dir="./cache")

        # Determine the device (GPU or CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def _preprocess_frames(self, frames):
        """
        Apply preprocessing to each frame using AutoImageProcessor.
        Resize frames to the required size and normalize them.
        """
        # Resize frames and return tensors
        inputs = self.image_processor(
            frames, return_tensors="pt", size=(224, 224))
        return inputs

    def _video_to_frames(self, video_path, target_frame_count):
        """
        Split video into frames using OpenCV.
        Dynamically select the number of frames based on video length and the required frame count.
        """
        vidcap = cv2.VideoCapture(video_path)
        # Total number of frames in the video
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate how many frames to skip for target_frame_count
        frame_skip = max(1, total_frames / target_frame_count)

        frames = []
        count = 0
        success, image = vidcap.read()
        while success:
            if count > frame_skip * len(frames):
                # Resize the frame using OpenCV if necessary
                image_resized = cv2.resize(image, (224, 224))
                frames.append(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            success, image = vidcap.read()
            count += 1

        vidcap.release()

        # If the number of selected frames is less than target_frame_count,
        # apply padding (add empty frames)
        while len(frames) < target_frame_count:
            # Add empty frame
            frames.append(torch.zeros(
                (224, 224, 3), dtype=torch.uint8).numpy())

        return frames

    def extract_features(self, video_path: str, target_frame_count: int = 16) -> torch.Tensor:
        """
        Extract features using the VideoMAE model.
        Dynamically select the number of frames based on video length.
        """
        frames = self._video_to_frames(video_path, target_frame_count)

        if not frames:
            raise ValueError("Failed to extract frames from video")

        # Preprocess the frames and resize to 224x224
        inputs = self._preprocess_frames(frames)

        # Move tensors to GPU or CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features using the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get embeddings
        embeddings = outputs.last_hidden_state  # or another output layer if needed
        return embeddings.mean(dim=1)
