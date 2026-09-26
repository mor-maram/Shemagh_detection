import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
#import timm

from shemagh_detection import (
    SystemConfig,
    ShemaghClassificationDataset,
    get_classification_transforms
)

def evaluate_saved_model():
    print("=" * 60)
    print("      Evaluating ConvNeXt Classification Accuracy")
    print("=" * 60)

    cfg = SystemConfig()
    device = cfg.DEVICE
    model_path = cfg.CONVNEXT_OUTPUT_DIR / f'convnext_fold{cfg.CONVNEXT_ACTIVE_FOLD}_best.pth'

    if not model_path.exists():
        print(f"Error: Model file not found at: {model_path}")
        return

    print(f"Loading trained model from: {model_path}")
    print(f"Device: {device}")

    # Load original training CSV
    df = pd.read_csv(cfg.TRAIN_CSV)
    df['path'] = df['filename'].apply(lambda fn: str(cfg.TRAIN_IMG_DIR / fn))

    # Re-create the exact same StratifiedKFold split used in training
    skf = StratifiedKFold(
        n_splits=cfg.CONVNEXT_N_FOLDS,
        shuffle=True,
        random_state=cfg.RANDOM_SEED
    )

    val_df = None
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(df, df['right_place'])):
        if fold_num == cfg.CONVNEXT_ACTIVE_FOLD:
            val_df = df.iloc[val_idx].reset_index(drop=True)
            break

    print(f"Validation dataset size: {len(val_df)} images")
    print(f"Class distribution in validation set: {val_df['right_place'].value_counts().to_dict()}")

    val_dataset = ShemaghClassificationDataset(
        val_df,
        transform=get_classification_transforms('val', cfg)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.CONVNEXT_BATCH,
        shuffle=False,
        num_workers=0
    )

    # Initialize model architecture and load saved weights
    model = timm.create_model(
        cfg.CONVNEXT_MODEL_NAME,
        pretrained=False,
        num_classes=cfg.CONVNEXT_NUM_CLASSES
    )
    
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("\nRunning inference on validation samples...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS")
    print("=" * 60)
    print(f"  >>> Accuracy:  {acc * 100:.2f}%  ({sum(p == l for p, l in zip(all_preds, all_labels))}/{len(all_labels)} correct)")
    print(f"  >>> F1-Score:  {f1 * 100:.2f}%")
    print(f"  >>> Precision: {precision * 100:.2f}%")
    print(f"  >>> Recall:    {recall * 100:.2f}%")
    print("=" * 60)

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Incorrect (0)', 'Correct (1)'], digits=4))

    print("Confusion Matrix:")
    print("                Predicted: 0   Predicted: 1")
    print(f"Actual 0 (Incorrect):   {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual 1 (Correct):     {cm[1][0]:<14} {cm[1][1]}")
    print("=" * 60)

if __name__ == '__main__':
    evaluate_saved_model()
