import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model.classificators.mlp_classifier.DataModule import VideoDataModule
from model.classificators.mlp_classifier.Classificator import MultiTaskClassifier


def train_model(
        video_meta_file, 
        categories_file, 
        tensor_dir, 
        batch_size=32, 
        max_epochs=10, 
        num_workers=4, 
        run_name="mlp-classifier-1", 
        hidden_layer_size=256,
        learning_rate=1e-3,
        checkpoints_dir="./checkpoints",
        model_save_path="./checkpoints/final_model.ckpt"
    ):
    # Инициализация логгера Wandb
    wandb_logger = WandbLogger(project="Video-tagging", name=run_name)
    
    # Создаем DataModule
    data_module = VideoDataModule(
        video_meta_file=video_meta_file,
        categories_file=categories_file,
        tensor_dir=tensor_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_val_test_split=(0.8, 0.2, 0.0)
    )
    
    data_module.setup()

    # Создаем модель
    model = MultiTaskClassifier(
        input_dim=1536, 
        num_classes=data_module.dataset.num_classes, 
        hidden_size=hidden_layer_size, 
        lr=learning_rate
    )

    # Настройка Wandb для отслеживания модели
    wandb_logger.watch(model, log_graph=False)
    
    # Коллбэк для сохранения наилучшей модели на основе валидационной точности
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",  # Ориентируемся на метрику валидационной точности
        dirpath=checkpoints_dir,  # Путь для сохранения чекпоинтов
        filename="best-checkpoint-{epoch:02d}-{val_iou:.2f}",  # Шаблон имени файла
        save_top_k=1,  # Сохраняем только лучший чекпоинт
        mode="max",  # Мы хотим максимизировать точность
        save_last=True,  # Также сохраняем последний чекпоинт
        verbose=True  # Показывать прогресс
    )

    # Определяем Trainer с чекпоинтами и ранней остановкой
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=4,
        callbacks=[checkpoint_callback]  # Добавляем коллбэки
    )
    
    # Запуск обучения
    trainer.fit(model, data_module)

    # Сохраняем последнюю версию модели
    trainer.save_checkpoint(model_save_path)


if __name__ == "__main__":
    # Запуск обучения
    train_model(
        video_meta_file="./train_dataset_tag_video/baseline/train_data_categories.csv",
        categories_file="./train_dataset_tag_video/baseline/IAB_tags.csv",
        tensor_dir="./embeddings_1536",
        batch_size=16,
        max_epochs=20,
        hidden_layer_size=256,
        checkpoints_dir="./checkpoints",
        model_save_path="./checkpoints/final_model.ckpt"
    )
