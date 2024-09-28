import torch
import lightning as pl
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import wandb
from tqdm import tqdm
import pandas as pd  # Добавляем импорт pandas для записи метрик в файл
import os  # Добавляем импорт os для работы с файловой системой


class MultiTaskLinearSVC(pl.LightningModule):
    def __init__(self, input_dim=1536, num_classes=611, learning_rate=1e-3, category_to_idx=None):
        super().__init__()
        self.training_data = []
        self.training_labels = []
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.classifier = OneVsRestClassifier(LinearSVC(max_iter=10000))
        self.scaler = StandardScaler()

        # Используем переданный параметр category_to_idx
        self.idx_to_category = {v: k for k, v in category_to_idx.items()}

        # Инициализация DataFrame для записи метрик
        self.metrics_df = pd.DataFrame(
            columns=['epoch', 'train_samples'])

        # Атрибут для хранения выходных данных теста
        self.test_outputs = []

    def forward(self, x):
        return x

    def setup(self, stage=None):
        # Fit the scaler with some initial data if available
        if self.training_data and self.training_labels:
            x = np.concatenate(self.training_data)
            self.scaler.fit(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        self.training_data.append(x)
        self.training_labels.append(y)

        loss = torch.tensor(0.0, requires_grad=True)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        if not self.training_data or not self.training_labels:
            print("No training data accumulated.")
            return

        x = np.concatenate(self.training_data)
        y = np.concatenate(self.training_labels)

        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape}")
        # Анализ меток
        label_counts = np.sum(y, axis=0)
        labels_to_exclude = []
        labels_not_present = []  # List to store labels not present in any example

        for idx, count in enumerate(label_counts):
            label_name = self.idx_to_category.get(idx + 1, "Unknown")
            # print(f"idx count label_name {idx} {count} {label_name}")
            if count == y.shape[0]:
                # print(f"idx count {idx} {count} in all exmpls")
                label_name = self.idx_to_category.get(idx + 1, "Unknown")
                # print(
                #     f"Label {idx} ({label_name}) is present in ALL training examples.")
                labels_to_exclude.append(idx)
            elif count == 0:
                # print(
                #     f"Label {idx} ({label_name}) is NOT in any training example.")
                labels_not_present.append(idx)
            # else:
                # print(f"Label {idx} ({label_name}) is good.")

        # Combine both lists of labels to exclude
        all_labels_to_exclude = labels_to_exclude + labels_not_present

        if all_labels_to_exclude:
            y = np.delete(y, all_labels_to_exclude, axis=1)

        # Get remaining labels
        remaining_labels_count = y.shape[1]
        print(f"Remaining labels count: {remaining_labels_count}")

        self.scaler.fit(x)
        x = self.scaler.transform(x)
        self.classifier.fit(x, y)
        print("fit classifier__________________")

        # Log metrics to wandb
        wandb.log({"train_samples": len(x)})

        # Save the labels to exclude for use in test_step
        self.labels_to_exclude = all_labels_to_exclude

        # Clear the accumulated data
        self.training_data = []
        self.training_labels = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # Ensure the scaler is fitted
        if not hasattr(self.scaler, 'mean_'):
            print("Fitting scaler during validation step.")
            self.scaler.fit(x)

        x = self.scaler.transform(x)

        if not hasattr(self.classifier, 'estimators_'):
            return None

        y_pred = self.classifier.predict(x)

        # Log metrics to wandb
        wandb.log({"val_samples": len(x)})

        # Append metrics to DataFrame
        self.metrics_df = self.metrics_df.append({
            'epoch': self.current_epoch,
            'train_samples': len(x)
        }, ignore_index=True)

        print(
            f"Validation - Epoch: {self.current_epoch}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        x = self.scaler.transform(x)
        y_hat = self.classifier.predict(x)

        # Удаляем метки, которые были исключены во время обучения
        if hasattr(self, 'labels_to_exclude') and self.labels_to_exclude:
            valid_indices = [
                idx for idx in self.labels_to_exclude if idx < y.shape[1]]
            if valid_indices:
                # print(f"Valid indices for deletion: {valid_indices}")
                y = np.delete(y, valid_indices, axis=1)

        # Сохраняем выходные данные для дальнейшей обработки
        self.test_outputs.append((y, y_hat))

        # Log metrics to wandb
        wandb.log({"test_samples": len(x)})

        # print(f"Test - Samples: {len(x)}")

    def on_test_epoch_end(self):
        # Обработка всех выходных данных после завершения тестовой эпохи
        y_true = np.concatenate([output[0] for output in self.test_outputs])
        y_pred = np.concatenate([output[1] for output in self.test_outputs])

        # Calculate IoU-like accuracy
        iou_accuracy = self.calculate_iou_accuracy(y_true, y_pred)
        print(f"\n\n\n IoU-like Accuracy: {iou_accuracy}\n\n\n")

        # Очищаем выходные данные после обработки
        self.test_outputs = []

    def on_epoch_end(self):
        # Save metrics to file
        self.metrics_df.to_csv("accur.csv", index=False)

        # Print all metrics to console
        print("Metrics DataFrame:")
        print(self.metrics_df)

    def configure_optimizers(self):
        return None

    def predict_for_video(self, video_id, tensor_dir):
        tensor_path = os.path.join(tensor_dir, f"{video_id}.pt")
        try:
            tensor = torch.load(tensor_path)
            tensor = tensor.view(tensor.shape[1])
        except Exception as e:
            print(f"Error loading tensor for video_id {video_id}: {str(e)}")
            return None

        tensor = tensor.numpy().reshape(1, -1)
        tensor = self.scaler.transform(tensor)
        predictions = self.classifier.predict(tensor)

        # Преобразование индексов в теги
        predicted_tags = [self.idx_to_category[idx + 1]
                          for idx, val in enumerate(predictions[0]) if val == 1]
        return predicted_tags

    def calculate_iou_accuracy(self, y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum(axis=1)
        union = np.logical_or(y_true, y_pred).sum(axis=1)
        iou = intersection / union
        return np.mean(iou)
