import os
import random
import pydicom

from sklearn.metrics import roc_auc_score, accuracy_score

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torchvision import models

from modules.helpers import *

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

NUM_WORKERS = 4
EPOCHS      = 50

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class ImageDatasetEMBED(Dataset):
    def __init__(self, meta_csv, transform=None):
        self.SOURCE = "sample_data"
        self.info = pd.read_csv(meta_csv)

        self.transform = transform
        self.image_paths = self.info['file_name'].tolist()
        self.labels = self.info['spot_label'].tolist()

        pos = sum(self.labels)
        print(f"[{os.path.basename(meta_csv)}] Loaded {len(self.image_paths)}, "
              f"pos(1)={pos}, neg(0)={len(self.labels)-pos}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.SOURCE, self.image_paths[idx])
        if '.dcm' in img_path:
            image = dicom_to_8bit_pil(img_path)
        else:
            image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return img_path, image, torch.tensor(self.labels[idx], dtype=torch.float32)

    # ── class-imbalance weight for BCEWithLogitsLoss ──────────────────────
    def pos_weight(self) -> torch.Tensor:
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        w   = neg / pos if pos > 0 else 1.0
        return torch.tensor([w], dtype=torch.float32)


# ─────────────────────────────────────────────
#  models
# ─────────────────────────────────────────────            
def build_backbone(model_name: str, pretrained: bool = True, dropout_p: float = 0.5) -> nn.Module:
    weights = "DEFAULT" if pretrained else None

    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, 1)
        )

    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, 1)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, 1)
        )

    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.classifier.in_features, 1)
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return model


# ─────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────
class ImageClassifier(pl.LightningModule):
    """
    Binary classifier for spot-compression detection.

    Logs per-step and per-epoch:
      training_loss, val_loss, val_acc, val_auc, collapse_flg
    """

    def __init__(
        self,
        model_name: str,
        lr: float,
        weight_decay: float,
        pos_weight: torch.Tensor,
        pretrained: bool = True,
        dropout_p: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        self.model      = build_backbone(model_name, pretrained, dropout_p=dropout_p)
        self.loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lr         = lr
        self.weight_decay = weight_decay

        # buffers accumulated per validation epoch
        self._val_preds:  list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []
        
        
    def forward(self, x):
        return self.model(x)

    # ── callbacks ─────────────────────────────────────────────────────────
    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            verbose=True,
        )
        checkpoint = ModelCheckpoint(
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_auc:.4f}",
            verbose=True,
        )
        return [early_stop, checkpoint]

    # ── optimiser + scheduler ─────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            #weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5) #mode min (default) or  max?
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc",
            },
        }

    # ── training ──────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        X, y    = batch
        logits  = self.model(X).squeeze(1) #squeeze to match BCEWithLogitsLoss shape
        loss    = self.loss_fn(logits, y)

        # collapse flag: number of unique predicted values (sanity check)
        collapse_flg = torch.unique(torch.sigmoid(logits).detach()).size(0) 

        self.log("training_loss", loss,         prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("collapse_flg", float(collapse_flg), on_epoch=True, sync_dist=True)
        return loss

    # ── validation ────────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        X, y   = batch
        logits = self.model(X).squeeze(1) #squeeze to match BCEWithLogitsLoss shape
        loss   = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits).cpu().numpy() 
        self._val_preds.append(probs)
        self._val_labels.append(y.cpu().numpy())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds  = np.concatenate(self._val_preds)
        labels = np.concatenate(self._val_labels)

        acc = accuracy_score(labels, (preds > 0.5).astype(int))
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.0  # only one class present in this epoch

        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        
        print(f"\n─── Validation Epoch {self.current_epoch} ───")
        print(f"Val Acc: {acc:.4f}  |  AUC: {auc:.4f}")

        # Store best-epoch validation preds for plotting
        if not hasattr(self, '_best_val_auc') or auc > self._best_val_auc:
            self._best_val_auc          = auc
            self._best_val_preds_final  = preds
            self._best_val_labels_final = labels

        self._val_preds  = []
        self._val_labels = []