import lightning as pl
from classificators.svm_classifier.Classificator import MultiTaskLinearSVC
from classificators.svm_classifier.DataModule import VideoDataModule


def main():
    # Указываем пути к файлам данных
    video_meta_file = "./train_dataset_tag_video/baseline/train_data_categories.csv"
    categories_file = "./train_dataset_tag_video/baseline/IAB_tags.csv"
    tensor_dir = "./embeddings_1536"

    # Инициализируем DataModule
    data_module = VideoDataModule(
        video_meta_file, categories_file, tensor_dir, batch_size=32)

    # Инициализируем SVM-классификатор
    model = MultiTaskLinearSVC(input_dim=1536, num_classes=611)

    # Используем Trainer для обучения модели
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
