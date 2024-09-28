import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, video_meta_df, categories_df, tensor_dir):
        self.video_meta_df = video_meta_df

        def filter_fn(row):
            return os.path.exists(
                os.path.join(
                    tensor_dir,
                    row['video_id'] + ".pt"
                )
            )
        m = self.video_meta_df.apply(filter_fn, axis=1)
        self.video_meta_df = self.video_meta_df[m]

        self.categories_df = categories_df
        self.tensor_dir = tensor_dir

        # Объединяем все уровни категорий
        self.categories_df['full_category'] = self.categories_df.apply(
            lambda row: ': '.join(filter(lambda x: str(x) != 'nan', [row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']])).strip().lower(), axis=1)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(
            self.categories_df['full_category'])}
        self.num_classes = len(self.category_to_idx)

        # Проверка иерархии тегов
        print("Category to index mapping:")
        for category, idx in self.category_to_idx.items():
            print(f"{category}: {idx}")

    def __len__(self):
        return len(self.video_meta_df)

    def __getitem__(self, idx):
        video_info = self.video_meta_df.iloc[idx]
        video_id = video_info['video_id']
        tensor_path = os.path.join(self.tensor_dir, f"{video_id}.pt")

        try:
            tensor = torch.load(tensor_path)
            tensor = tensor.view(tensor.shape[1])
        except Exception as e:
            print(f"Error loading tensor for video_id {video_id}: {str(e)}")
            return None

        # Обработка меток для мультиклассовой классификации
        labels = torch.zeros(self.num_classes)
        tags = video_info['tags']

        if isinstance(tags, str):
            tag_list = [tag.strip().lower() for tag in tags.split(',')]
            for tag in tag_list:
                # Разбиваем по уровню
                level_parts = tag.split(':')
                # Учитываем, что может быть 1, 2 или 3 уровня
                if len(level_parts) == 1:
                    full_tag = level_parts[0]
                elif len(level_parts) == 2:
                    full_tag = f"{level_parts[0]}: {level_parts[1]}"
                elif len(level_parts) == 3:
                    full_tag = f"{level_parts[0]}: {level_parts[1]}: {level_parts[2]}"

                if full_tag in self.category_to_idx:
                    labels[self.category_to_idx[full_tag]] = 1
                else:
                    print(f"Tag '{full_tag}' not found in category_to_idx")
        else:
            print(
                f"Warning: tags for video_id {video_id} is not a string. Value: {tags}")

        return tensor.numpy(), labels.numpy()


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, video_meta_file, categories_file, tensor_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.video_meta_file = video_meta_file
        self.categories_file = categories_file
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        video_meta_df = pd.read_csv(self.video_meta_file)
        categories_df = pd.read_csv(self.categories_file)
        self.dataset = VideoDataset(
            video_meta_df, categories_df, self.tensor_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)