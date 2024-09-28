import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar
from classificators.svm_classifier.Classificator import MultiTaskLinearSVC
from classificators.svm_classifier.DataModule import VideoDataModule
import wandb


def main():
    # Инициализация wandb
    wandb.init(project="svm_video_tagging")

    # Указываем пути к файлам данных
    video_meta_file = "./train_dataset_tag_video/baseline/train_data_categories.csv"
    categories_file = "./train_dataset_tag_video/baseline/IAB_tags.csv"
    tensor_dir = "./embeddings_1536"

    # Инициализируем DataModule
    data_module = VideoDataModule(
        video_meta_file, categories_file, tensor_dir, batch_size=32)

    # Инициализируем SVM-классификатор
    model = MultiTaskLinearSVC(input_dim=1536, num_classes=611)

    # Инициализируем WandbLogger
    wandb_logger = WandbLogger()

    # Инициализируем TQDMProgressBar
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Используем Trainer для обучения модели
    trainer = pl.Trainer(
        max_epochs=1, logger=wandb_logger, callbacks=[progress_bar])
    trainer.fit(model, datamodule=data_module)

    # Завершаем wandb
    wandb.finish()


if __name__ == '__main__':
    main()
