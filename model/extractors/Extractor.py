from extractors.video_extractor.VideoFeatureExtractor import VideoFeatureExtractor
from extractors.text_extractor.TextFeatureExtractor import TextFeatureExtractor
import os
from tqdm import tqdm
import torch


class Extractor:
    def __init__(
            self, 
            path_to_video_folder, 
            path_to_output_folder,
            use_video_embeddings=True,
            use_text_embeddings=True,
        ):
        self.path_to_video_folder = path_to_video_folder
        self.path_to_output_folder = path_to_output_folder

        self.use_video_embeddings = use_video_embeddings
        self.use_text_embeddings = use_text_embeddings

        if use_video_embeddings:
            self.video_extractor = VideoFeatureExtractor()
        if use_text_embeddings:
            self.text_extractor = TextFeatureExtractor()
    
    def __call__(self, video_id, title, description, save=True, show_progress=False):
        
        if show_progress:
            pbar = tqdm(total=(
                int(self.use_video_embeddings) +
                int(self.use_text_embeddings) +
                int(save)
            ), desc="Extracting features")

        embeddings = []

        # Extracting video features
        if self.use_video_embeddings:
            if show_progress:
                pbar.set_description("Extracting video features")
            video_embeddings = self.video_extractor.extract_features(os.path.join(self.path_to_video_folder, video_id + ".mp4"))
            embeddings.append(video_embeddings)
            if show_progress:
                pbar.update(1)
        
        # Extracting text features
        if self.use_text_embeddings:
            if show_progress:
                pbar.set_description("Extracting text features")
            text_embeddings = self.text_extractor.extract_features(f"{title} {description}")
            embeddings.append(text_embeddings)
            if show_progress:
                pbar.update(1)

        # Concatenate embeddings
        embeddings = torch.cat(embeddings, dim=1)

        # Saving embeddings
        if save:
            if show_progress:
                pbar.set_description("Saving embeddings")
            dimension = embeddings.shape[1]
            os.makedirs(self.path_to_output_folder + "_" + str(dimension), exist_ok=True)
            output_file = os.path.join(self.path_to_output_folder + "_" + str(dimension), video_id + ".pt")
            torch.save(embeddings, output_file)
            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()
        return embeddings