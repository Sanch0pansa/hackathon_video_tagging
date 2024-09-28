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

    # Инициализируем DataModule с уменьшенной выборкой
    data_module = VideoDataModule(
        video_meta_file, categories_file, tensor_dir, batch_size=32, sample_size=100)

    # Вызовем setup для инициализации dataset
    data_module.setup(stage='fit')

    # Получаем category_to_idx из DataModule
    category_to_idx = data_module.get_category_to_idx()

    # Инициализируем SVM-классификатор
    model = MultiTaskLinearSVC(
        input_dim=1536, num_classes=611, category_to_idx=category_to_idx)

    # Инициализируем WandbLogger
    wandb_logger = WandbLogger()

    # Инициализируем TQDMProgressBar
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Используем Trainer для обучения модели
    trainer = pl.Trainer(
        max_epochs=1, logger=wandb_logger, callbacks=[progress_bar])
    trainer.fit(model, datamodule=data_module)

    # Запуск тестирования модели
    trainer.test(model, datamodule=data_module)

    # Пример использования модели для предсказания меток для заданного video_id
    video_id = "cf3ef0b2d6227ad372a9b7dcb6cb2df3"  # Замените на реальный video_id
    predicted_tags = model.predict_for_video(video_id, tensor_dir)
    if predicted_tags is not None:
        print(f"Predicted tags for video_id {video_id}: {predicted_tags}")
    # answer is: cf3ef0b2d6227ad372a9b7dcb6cb2df3,
    # Смехмашина | Выпуск 13,"Подвели итоги розыгрыша. Поздравляем счастливого обладателя новенького IPhone 13 Pro, им стал @sssr198787  Ссылки на наши каналы: Антон Протеинов - / Люди у которых клюёт - / Пацанский клининг - / Пойдём отойдём - / ПОЛУПАНОВЫ - / Тот Самый Мент - / Тяпа - / Шляпа - / Максим Народный - /  Новый выпуск «Смехмашины», в котором принимают участие Сундук, Максим и Егор. В этом выпуске участники шоу будут в роли водителей такси! Вы представляете?! Перед парнями стоит непростая задача, помимо того, что бы смешно рассказать анекдот, нужно довести пассажира до определенной точки. Кто победит в этом непростом выпуске? Вы узнаете прямо сейчас!",
    # Массовая культура: Юмор и сатира

    # Завершаем wandb
    wandb.finish()


if __name__ == '__main__':
    main()
