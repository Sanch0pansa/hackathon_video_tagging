import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import os

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
            lambda row: ': '.join(filter(lambda x: str(x) != 'nan', [row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']])),
            axis=1
        )
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories_df['full_category'])}
        self.num_classes = len(self.category_to_idx)

    def __len__(self):
        return len(self.video_meta_df)

    def __getitem__(self, idx):
        video_info = self.video_meta_df.iloc[idx]
        video_id = video_info['video_id']
        tensor_path = os.path.join(self.tensor_dir, f"{video_id}.pt")
        tensor = torch.load(tensor_path, weights_only=True)
        tensor = tensor.view(tensor.shape[1])

        # Размечаем метки для мультиклассовой классификации
        tags = (str(video_info['tags']) if str(video_info['tags']) != "nan" else "").split(', ')
        labels = torch.zeros(self.num_classes)
        for tag in tags:
            tag = tag.strip()
            if tag in self.category_to_idx:
                labels[self.category_to_idx[tag]] = 1
        return tensor, labels

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, video_meta_file, categories_file, tensor_dir, batch_size=32, num_workers=4, train_val_test_split=(0.7, 0.15, 0.15)):
        super().__init__()
        self.video_meta_file = video_meta_file
        self.categories_file = categories_file
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.dataset = None

    def setup(self, stage=None):
        # Читаем данные
        video_meta_df = pd.read_csv(self.video_meta_file)
        categories_df = pd.read_csv(self.categories_file)

        # Создаем основной датасет
        dataset = VideoDataset(video_meta_df, categories_df, self.tensor_dir)
        self.dataset = dataset

        # Разбиваем на тренировочную, валидационную и тестовую выборки
        train_size = int(self.train_val_test_split[0] * len(dataset))
        val_size = int(self.train_val_test_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size  # Оставшееся количество на тест

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # фиксируем seed для воспроизводимости
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ ==  '__main__':
    video_meta_df = pd.read_csv('./train_dataset_tag_video/baseline/train_data_categories.csv')
    categories_df = pd.read_csv('./train_dataset_tag_video/baseline/IAB_tags.csv')
    dataset = VideoDataset(video_meta_df, categories_df, 'embeddings_1536')
    print(dataset[0])