import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import re  # Добавляем импорт re для использования регулярных выражений
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, video_meta_df, categories_df, tensor_dir):
        self.video_meta_df = video_meta_df
        self.categories_df = categories_df  # Сохраняем переданный DataFrame
        self.tensor_dir = tensor_dir

        # Фильтрация видео на основе существующих тензоров
        self.video_meta_df = self.video_meta_df[
            self.video_meta_df['video_id'].apply(
                lambda x: os.path.exists(os.path.join(tensor_dir, f"{x}.pt"))
            )
        ]

        # Обработка тегов из файла иерархии
        self.category_to_idx = self.process_categories()

        # Список всех возможных меток для проверки наличия в каждом видео
        self.all_labels = set(self.category_to_idx.values())

    def process_categories(self):
        category_mapping = {}
        idx = 1  # Начинаем индексацию с 1

        for _, row in self.categories_df.iterrows():
            # Проверяем наличие и тип данных в Уровень 3 (iab)
            if isinstance(row['Уровень 3 (iab)'], str) and row['Уровень 3 (iab)'].strip() != '':
                tag = row['Уровень 3 (iab)'].strip().lower()
            elif isinstance(row['Уровень 2 (iab)'], str) and row['Уровень 2 (iab)'].strip() != '':
                tag = row['Уровень 2 (iab)'].strip().lower()
            elif isinstance(row['Уровень 1 (iab)'], str) and row['Уровень 1 (iab)'].strip() != '':
                tag = row['Уровень 1 (iab)'].strip().lower()
            else:
                continue  # Пропускаем строки без заполненных полей

            category_mapping[tag] = idx
            idx += 1

        # self.log_category_mapping(category_mapping)
        return category_mapping

    def log_category_mapping(self, category_mapping):
        with open("category_mapping_log.txt", "w") as log_file:
            for tag, idx in category_mapping.items():
                log_file.write(f"{tag}: {idx}\n")

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
        labels = torch.zeros(len(self.category_to_idx))
        tags = video_info['tags']

        if isinstance(tags, str):
            tag_list = [tag.strip().lower() for tag in re.split(
                r',|:', tags)]  # Разделение по запятой и двоеточию
            for tag in tag_list:
                if tag in self.category_to_idx:
                    labels[self.category_to_idx[tag] - 1] = 1
                else:
                    print(f"Tag '{tag}' not found in category_to_idx")
        else:
            print(
                f"Warning: tags for video_id {video_id} is not a string. Value: {tags}")
        return tensor.numpy(), labels.numpy()

    def get_category_to_idx(self):
        return self.category_to_idx


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, video_meta_file, categories_file, tensor_dir, batch_size=32, num_workers=4, sample_size=None):
        super().__init__()
        self.video_meta_file = video_meta_file
        self.categories_file = categories_file
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_size = sample_size  # Добавляем параметр sample_size

    def setup(self, stage=None):
        video_meta_df = pd.read_csv(self.video_meta_file)
        categories_df = pd.read_csv(self.categories_file)

        # Уменьшаем выборку данных, если указано sample_size
        if self.sample_size:
            video_meta_df = video_meta_df.sample(n=self.sample_size)

        self.dataset = VideoDataset(
            video_meta_df, categories_df, self.tensor_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_category_to_idx(self):
        return self.dataset.get_category_to_idx()
