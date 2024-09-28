import torch
import lightning as pl
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import wandb
from tqdm import tqdm


class MultiTaskLinearSVC(pl.LightningModule):
    def __init__(self, input_dim=1536, num_classes=611, learning_rate=1e-3):
        super().__init__()
        self.training_data = []
        self.training_labels = []
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.classifier = OneVsRestClassifier(LinearSVC(max_iter=10000))
        self.scaler = StandardScaler()

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
        for idx, count in enumerate(label_counts):
            if count == y.shape[0]:
                print(f"Label {idx} is present in all training examples.")

        self.scaler.fit(x)
        x = self.scaler.transform(x)
        self.classifier.fit(x, y)

        # Log metrics to wandb
        wandb.log({"train_samples": len(x)})

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
            self.log('val_f1_score', 0.0)
            return None

        y_pred = self.classifier.predict(x)
        f1 = f1_score(y, y_pred, average='micro')
        self.log('val_f1_score', f1)
        wandb.log({"val_f1_score": f1})
        return f1

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        x = self.scaler.transform(x)
        y_hat = self.classifier.predict(x)

        test_f1 = f1_score(y, y_hat, average='micro')
        self.log('test_f1_score', test_f1)
        wandb.log({"test_f1_score": test_f1})
        return test_f1

    def configure_optimizers(self):
        return None
