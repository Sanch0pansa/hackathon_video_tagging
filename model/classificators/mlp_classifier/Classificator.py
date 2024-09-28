import torch.nn as nn
import torch
import torchmetrics
import lightning as pl
import torchmetrics.classification


class JaccardLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Args:
            eps (float): Small value to avoid division by zero.
        """
        super(JaccardLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        Compute Jaccard Loss.
        
        Args:
            preds (torch.Tensor): Model predictions of shape (batch_size, num_classes).
                                  Assumed to be probabilities in the range [0, 1].
            targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
                                    Binary values (0 or 1).
        
        Returns:
            loss (torch.Tensor): Jaccard Loss value.
        """
        # Apply sigmoid if preds are logits, else you can skip this step.
        preds = torch.sigmoid(preds)  # Keep preds in the range [0, 1] for smooth computation
        
        # Intersection and union (No need for hard thresholding)
        intersection = (preds * targets).sum(dim=1)  # Sum over classes
        union = (preds + targets - preds * targets).sum(dim=1)  # Sum over classes
        
        # Jaccard index (IoU) for each sample in the batch
        iou = (intersection + self.eps) / (union + self.eps)  # Avoid division by zero
        
        # Jaccard loss (1 - IoU)
        loss = 1 - iou
        
        # Return the mean loss over the batch
        return loss.mean()
    

class CustomIoU:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, preds, targets):
        """
        Compute IoU score between predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions of shape (batch_size, num_classes).
                                  Assumed to be in the range [0, 1] (probabilities).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
                                    Binary (0 or 1).
        
        Returns:
            iou_score (float): The average IoU score across all samples and classes.
        """
        # Binary thresholding of predictions
        preds = (preds > self.threshold).float()
        
        # Intersection: (preds == 1 and targets == 1)
        intersection = (preds * targets).sum(dim=1)  # Sum over batch
        
        # Union: (preds == 1 or targets == 1)
        union = (preds + targets).clamp(0, 1).sum(dim=1)  # Sum over batch
        
        # IoU for each class
        iou_per_class = intersection / (union + 1e-6)  # Avoid division by zero
        
        # Mean IoU across all classes
        mean_iou = iou_per_class.mean().item()
        
        return mean_iou

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
        self.train_accuracy = torchmetrics.classification.MultilabelAccuracy(num_labels=num_classes, average='micro')
        self.train_precision = torchmetrics.classification.MultilabelPrecision(num_labels=num_classes, average='micro')
        self.train_iou = torchmetrics.classification.MultilabelJaccardIndex(num_labels=num_classes, average='macro')

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
        criterion = JaccardLoss()
        loss = criterion(logits, y)
        
        # Predictions (after sigmoid to get probabilities)
        preds = torch.sigmoid(logits)
        
        # Log loss and metrics
        custom_iou = CustomIoU(threshold=0.5)
        iou_value = custom_iou(preds, y)

        # Log IoU and loss
        self.log('train_iou', iou_value, prog_bar=True)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(preds, y), prog_bar=True)
        self.log('train_precision', self.train_precision(
            preds, y), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass
        logits = self(x)
        criterion = JaccardLoss()
        loss = criterion(logits, y)
        
        # Predictions
        preds = torch.sigmoid(logits)
        
        custom_iou = CustomIoU(threshold=0.5)
        iou_value = custom_iou(preds, y)

        # Log IoU and loss
        self.log('val_iou', iou_value, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(preds, y), prog_bar=True)
        self.log('val_precision', self.val_precision(preds, y), prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
