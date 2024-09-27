from extractors.Extractor import Extractor
import pandas as pd
import os
from tqdm import tqdm

dataset = pd.read_csv("./train_dataset_tag_video/baseline/train_data_categories.csv")

extractor = Extractor(
    "./train_dataset_tag_video/videos/",
    "./embeddings",
)

for _, row in tqdm(dataset.iterrows(), desc="Extracting features from dataset", total=len(dataset)):
    video_id = row["video_id"]
    title = row["title"]
    description = row["description"]

    if os.path.exists(os.path.join("./train_dataset_tag_video/videos/", video_id + ".mp4")):
        extractor(video_id, title, description, save=True)