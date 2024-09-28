import torch.nn as nn
import torch
import torchmetrics
import lightning as pl
import torchmetrics.classification

# MLP Model for Multi-Label Classification


class MultiTaskClassifier(pl.LightningModule):
    def __init__(self, input_dim=1536, num_classes=611, hidden_size=256, lr=1e-4):
        super(MultiTaskClassifier, self).__init__()
        self.lr = lr

        # Define MLP layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        # Loss function (Binary Cross-Entropy with Logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = torchmetrics.classification.MultilabelAccuracy(
            num_labels=num_classes, average='micro')
        self.train_precision = torchmetrics.classification.MultilabelPrecision(
            num_labels=num_classes, average='micro')
        self.train_iou = torchmetrics.classification.MultilabelJaccardIndex(
            num_labels=num_classes)

        self.val_accuracy = torchmetrics.classification.MultilabelAccuracy(
            num_labels=num_classes, average='micro')
        self.val_precision = torchmetrics.classification.MultilabelPrecision(
            num_labels=num_classes, average='micro')
        self.val_iou = torchmetrics.classification.MultilabelJaccardIndex(
            num_labels=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)

        # Predictions (after sigmoid to get probabilities)
        preds = torch.sigmoid(logits)
        self.log('train_iou', self.train_iou(preds, y), prog_bar=True)

        preds = (preds > 0.5).float()  # Binary predictions

        # Log loss and metrics
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(preds, y), prog_bar=True)
        self.log('train_precision', self.train_precision(
            preds, y), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)

        # Predictions
        preds = torch.sigmoid(logits)
        self.log('val_iou', self.val_iou(preds, y), prog_bar=True)

        preds = (preds > 0.5).float()

        # Log loss and metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(preds, y), prog_bar=True)
        self.log('val_precision', self.val_precision(preds, y), prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
