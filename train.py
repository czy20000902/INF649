import argparse
import os
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple

import albumentations as A
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from albumentations.pytorch import ToTensorV2
from colorama import Back, Fore, Style
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

from model.mobileone import MobileOne
from model.MemotionModel import MemotionModel
from utils.dataset import MemotionDataset

import warnings

warnings.simplefilter('ignore')

# from PIL import ImageFile
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(47)
np.random.seed(47)

# logger = logging.getLogger(__name__)
logging.basicConfig(filename='training.log', level=logging.INFO)

parser = argparse.ArgumentParser(description='Your description here')

# Define arguments
parser.add_argument('--task', type=int, default=3)
# root = 'E:/Dataset/memotion_dataset_7k/'
parser.add_argument('--root', type=str, default='/users/eleves-a/2022/zhaoyang.chen/memotion_dataset_7k/')
parser.add_argument('--save_path', type=str, default='checkpoints')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224])
parser.add_argument('--epochs', type=int, default=50)

# Optimizers
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=1e-3)

# Scheduler
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--lr_gamma', type=float, default=0.1)

# Config
parser.add_argument('--num_classes', type=int, default=5)

# Device
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


@torch.no_grad()
def validate(
        model: nn.Module, dataloader: DataLoader
) -> Tuple[float, dict]:
    model.eval()
    dataset_size = 0
    running_loss = 0
    losses = {}

    criterion = nn.CrossEntropyLoss()

    accuracy_metric_humour = Accuracy(task="multiclass", num_classes=4)
    accuracy_metric_sarcasm = Accuracy(task="multiclass", num_classes=4)
    accuracy_metric_motivational = Accuracy(task="multiclass", num_classes=4)
    accuracy_metric_offensive = Accuracy(task="multiclass", num_classes=2)
    accuracy_metric_overall_sentiment = Accuracy(task="multiclass", num_classes=5)

    val_scores = defaultdict(list)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(valid) ")
    for step, batch in pbar:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        label_humour = batch['label_humour']
        label_sarcasm = batch['label_sarcasm']
        label_offensive = batch['label_offensive']
        label_motivational = batch['label_motivational']
        label_overall_sentiment = batch['label_overall_sentiment']
        yHat = model.forward(**batch)

        losses['loss_humour'] = criterion(yHat[0], label_humour)
        losses['loss_sarcasm'] = criterion(yHat[1], label_sarcasm)
        losses['loss_motivational'] = criterion(yHat[2], label_offensive)
        losses['loss_offensive'] = criterion(yHat[3], label_motivational)
        losses['loss_overall_sentiment'] = criterion(yHat[4], label_overall_sentiment)

        loss = sum(l for l in losses.values())
        running_loss += loss.item() * label_overall_sentiment.shape[0]
        dataset_size += label_overall_sentiment.shape[0]

        epoch_loss = running_loss / dataset_size

        accuracy = 0

        accuracy += accuracy_metric_humour(torch.argmax(yHat[0], axis=1).cpu(), label_humour.cpu())
        accuracy += accuracy_metric_sarcasm(torch.argmax(yHat[1], axis=1).cpu(), label_sarcasm.cpu())
        accuracy += accuracy_metric_motivational(torch.argmax(yHat[2], axis=1).cpu(), label_offensive.cpu())
        accuracy += accuracy_metric_offensive(torch.argmax(yHat[3], axis=1).cpu(), label_motivational.cpu())
        accuracy += accuracy_metric_overall_sentiment(torch.argmax(yHat[4], axis=1).cpu(), label_overall_sentiment.cpu())
        


        val_scores["accuracy"].append(accuracy / 5)

        # wandb.log(
        #     {
        #         "valid/loss": epoch_loss,
        #         "valid/accuracy": accuracy,
        #         "valid/precision": precision,
        #         "valid/recall": recall,
        #         "valid/auroc": auroc,
        #         "valid/f1": f1,
        #     },
        #     step=step,
        # )

    return epoch_loss, val_scores


if __name__ == "__main__":
    print('#' * 16)
    print(f'### Training ###')
    print('#' * 16)

    # run = wandb.init(project='multimodal-sentiment-analysis')
    run = wandb.init(project='multimodal-humour-analysis')
    # run = wandb.init(project='multimodal-sentiment-analysis-resnet')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_transforms = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1]),
        ToTensorV2(),
    ])
    test_transforms = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1]),
        ToTensorV2(),
    ])

    # df = pd.read_csv('folds.csv')
    # train_df = df[df['kfold'] != fold].reset_index(drop=True)
    # valid_df = df[df['kfold'] == fold].reset_index(drop=True)

    train_dataset = MemotionDataset(split="train", root=args.root, transform=train_transforms)
    valid_dataset = MemotionDataset(root=args.root, transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = MemotionModel(task=args.task).to(args.device)
    # model = mobileone(num_classes=args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=args.lr_gamma)

    best_loss = np.inf
    best_epoch = -1

    checkpoint = os.path.join(args.save_path, 'model_best.pth')
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint), strict=False)

    for epoch in range(args.epochs):
        print(f"\t\t\t\t########## EPOCH [{epoch + 1}/{args.epochs}] ##########")

        model.train()
        dataset_size = 0
        running_loss = 0
        losses = {}

        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            label_humour = batch['label_humour']
            label_sarcasm = batch['label_sarcasm']
            label_offensive = batch['label_offensive']
            label_motivational = batch['label_motivational']
            label_overall_sentiment = batch['label_overall_sentiment']
            yHat = model.forward(**batch)

            optimizer.zero_grad()

            losses['loss_humour'] = criterion(yHat[0], label_humour)
            losses['loss_sarcasm'] = criterion(yHat[1], label_sarcasm)
            losses['loss_motivational'] = criterion(yHat[2], label_offensive)
            losses['loss_offensive'] = criterion(yHat[3], label_motivational)
            losses['loss_overall_sentiment'] = criterion(yHat[4], label_overall_sentiment)

            loss = sum(l for l in losses.values())
            loss.backward()

            # losses.backward()
            optimizer.step()

            running_loss += loss.item() * label_overall_sentiment.shape[0]
            dataset_size += label_overall_sentiment.shape[0]

            # out = torch.argmax(yHat, axis=1)
            # accuracy = accuracy_metric(out.cpu(), labels.cpu())
            # precision = precision_metric(out.cpu(), labels.cpu())
            # recall = recall_metric(out.cpu(), labels.cpu())
            # auroc = auroc_metric(F.softmax(yHat, dim=1).cpu(), labels.cpu())
            # f1 = f1_metrics(out.cpu(), labels.cpu())
            current_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(loss=f"{loss.item():.5f}", current_lr=f"{current_lr:.5f}")

        epoch_loss = running_loss / dataset_size

        if scheduler is not None:
            scheduler.step()

        valid_loss, valid_scores = validate(
            model=model, dataloader=valid_dataloader
        )
        run.log(
            {
                "train_loss": epoch_loss,
                "valid_loss": valid_loss,
                "valid_accuracy": np.mean(valid_scores["accuracy"]),
                # "valid_precision": np.mean(valid_scores["precision"]),
                # "valid_recall": np.mean(valid_scores["recall"]),
                # "valid_auroc": np.mean(valid_scores["auroc"]),
                # "valid_f1": np.mean(valid_scores["f1"]),
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        print(
            f'Valid Accuracy: {np.mean(valid_scores["accuracy"]):.5f} | Valid Loss: {valid_loss:.5f}'
        )

        if valid_loss < best_loss:
            print(
                f"{Fore.GREEN}Validation Score Improved from {best_loss:.5f} to {valid_loss:.5f}"
            )
            best_epoch = epoch + 1
            best_loss = valid_loss

            torch.save(model.state_dict(), f"{args.save_path}/model_best.pth")
            print(f"MODEL SAVED!{Style.RESET_ALL}")

        torch.save(model.state_dict(), f"{args.save_path}/model_last.pth")

    run.finish()
