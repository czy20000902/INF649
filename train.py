import copy
import gc
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
from torchmetrics import AUROC, Accuracy, F1, Precision, Recall
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModel



from model.mobileone import MobileOne
from utils.dataset import MemotionDataset

# import warnings
# warnings.simplefilter('ignore')

# from PIL import ImageFile
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True


# logger = logging.getLogger(__name__)
logging.basicConfig(filename='training.log', level=logging.INFO)

class Config:
    debug = False  # set debug=False for Full Training
    root = 'E:/Dataset/memotion_dataset_7k/labels.csv'
    save_path = 'checkpoints'
    exp_name = "vit/sbert"
    model_name = "vit-sbert-multimodal"
    backbone = "google/vit-base-patch16-224+sentence-transformers/all-mpnet-base-v2-ep10"
    image_encoder = "google/vit-base-patch16-224"
    batch_size = 16
    img_size = [224, 224]
    epochs = 50

    # Optimizers
    optimizer = 'Adam'
    learning_rate = 3e-4
    rho = 0.9
    eps = 1e-6
    lr_decay = 0
    betas = (0.9, 0.999)
    alpha = 0.99

    # Scheduler
    min_lr = 1e-6
    warmup_epochs = 0
    lr_gamma = 0.1

    # Config
    num_folds = 5
    num_classes = 5

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


# Import wandb library for logging and tracking experiments

# Try to get the API key from Kaggle secrets
# try:
#     from kaggle_secrets import UserSecretsClient
#
#     user_secrets = UserSecretsClient()
#     api_key = user_secrets.get_secret("WANDB")
#     # Login to wandb with the API key
#     wandb.login(key=api_key)
#     # Set anonymous mode to None
#     anonymous = None
# except:
#     # If Kaggle secrets are not available, set anonymous mode to 'must'
#     anonymous = 'must'
#     # Login to wandb anonymously and relogin if needed
#     wandb.login(anonymous=anonymous, relogin=True)



class ImageEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(Config.image_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward(x)["pooler_output"]
        return x


class TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(Config.tokenizer)

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
        return x["pooler_output"]


class MemotionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = MobileOne(width_multipliers=[1,1,1,1])
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.alpha_img = torch.randn(size=(1,), requires_grad=True, device=Config.device)
        self.alpha_txt = torch.randn(size=(1,), requires_grad=True, device=Config.device)
        # self.fc1 = nn.Linear(768, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 5)
        self.linear = nn.Linear(512, 5)
        self.dropout = nn.Dropout(p=0.2)

    def forward(
            self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_out = self.image_encoder.forward(image)
        txt_out = self.text_encoder.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        print(self.alpha_txt.shape)
        print(txt_out.pooler_output.shape)
        print(self.alpha_img.shape)
        print(img_out.shape)
        wt_emb = self.alpha_txt * txt_out.pooler_output + self.alpha_img * img_out
        # x = self.fc1(self.dropout(wt_emb))
        # x = self.fc2(self.dropout(x))
        # return self.fc3(x)
        return self.linear(wt_emb)

def train(
        model: nn.Module,
        optimizer: optim,
        dataloader: DataLoader,
        scheduler=None,
) -> float:
    model.train()
    dataset_size = 0
    running_loss = 0

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(num_classes=Config.num_classes)
    precision_metric = Precision(num_classes=Config.num_classes)
    recall_metric = Recall(num_classes=Config.num_classes)
    auroc_metric = AUROC(num_classes=Config.num_classes)
    f1_metrics = F1(num_classes=Config.num_classes)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(train) ")
    for step, batch in pbar:
        batch = {k: v.to(Config.device) for k, v in batch.items()}
        labels = batch["label"]
        yHat = model.forward(**batch)

        optimizer.zero_grad()
        loss = criterion(yHat, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * labels.shape[0]
        dataset_size += labels.shape[0]

        epoch_loss = running_loss / dataset_size

        out = torch.argmax(yHat, axis=1)
        accuracy = accuracy_metric(out.cpu(), labels.cpu())
        precision = precision_metric(out.cpu(), labels.cpu())
        recall = recall_metric(out.cpu(), labels.cpu())
        auroc = auroc_metric(F.softmax(yHat, dim=1).cpu(), labels.cpu())
        f1 = f1_metrics(out.cpu(), labels.cpu())
        current_lr = optimizer.param_groups[0]["lr"]

        # wandb.log(
        #     {
        #         "train/loss": epoch_loss,
        #         "train/accuracy": accuracy,
        #         "train/precision": precision,
        #         "train/recall": recall,
        #         "train/auroc": auroc,
        #         "train/f1": f1,
        #         "train/current_lr": current_lr,
        #     },
        #     step=step,
        # )

        pbar.set_postfix(epoch_loss=f"{epoch_loss:.5f}", current_lr=f"{current_lr:.5f}")

    return epoch_loss


@torch.no_grad()
def validate(
        model: nn.Module, dataloader: DataLoader
) -> Tuple[float, dict]:
    model.eval()
    dataset_size = 0
    running_loss = 0

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(num_classes=Config.num_classes)
    precision_metric = Precision(num_classes=Config.num_classes)
    recall_metric = Recall(num_classes=Config.num_classes)
    auroc_metric = AUROC(num_classes=Config.num_classes)
    f1_metrics = F1(num_classes=Config.num_classes)

    val_scores = defaultdict(list)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(valid) ")
    for step, batch in pbar:
        batch = {k: v.to(Config.device) for k, v in batch.items()}
        labels = batch["label"]
        yHat = model.forward(**batch)

        loss = criterion(yHat, labels)

        running_loss += loss.item() * labels.shape[0]
        dataset_size += labels.shape[0]

        epoch_loss = running_loss / dataset_size

        out = torch.argmax(yHat, axis=1)
        accuracy = accuracy_metric(out.cpu(), labels.cpu())
        precision = precision_metric(out.cpu(), labels.cpu())
        recall = recall_metric(out.cpu(), labels.cpu())
        auroc = auroc_metric(F.softmax(yHat, dim=1).cpu(), labels.cpu())
        f1 = f1_metrics(out.cpu(), labels.cpu())

        val_scores["accuracy"].append(accuracy)
        val_scores["precision"].append(precision)
        val_scores["recall"].append(recall)
        val_scores["auroc"].append(auroc)
        val_scores["f1"].append(f1)

        wandb.log(
            {
                "valid/loss": epoch_loss,
                "valid/accuracy": accuracy,
                "valid/precision": precision,
                "valid/recall": recall,
                "valid/auroc": auroc,
                "valid/f1": f1,
            },
            step=step,
        )

    return epoch_loss, val_scores

if __name__ == "__main__":
    print('#' * 15)
    print(f'### Training ###')
    print('#' * 15)

    # run = wandb.init(project='multimodal-sentiment-analysis')

    train_transforms = A.Compose([
        A.Resize(height=Config.img_size[0], width=Config.img_size[1]),
        ToTensorV2(),
    ])
    test_transforms = A.Compose([
        A.Resize(height=Config.img_size[0], width=Config.img_size[1]),
        ToTensorV2(),
    ])

    # df = pd.read_csv('folds.csv')
    # train_df = df[df['kfold'] != fold].reset_index(drop=True)
    # valid_df = df[df['kfold'] == fold].reset_index(drop=True)

    train_dataset = MemotionDataset(split="train", root=Config.root, transform=train_transforms)
    valid_dataset = MemotionDataset(split="val", root=Config.root, transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=Config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=Config.batch_size, shuffle=False)

    model = MemotionModel().to(Config.device)
    # model = mobileone(num_classes=Config.num_classes).to(Config.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=Config.lr_gamma)


    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_epoch = -1

    for epoch in range(Config.epochs):
        gc.collect()
        print(f"\t\t\t\t########## EPOCH [{epoch + 1}/{Config.epochs}] ##########")
        train_loss = train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
        )
        valid_loss, valid_scores = validate(
            model=model, dataloader=valid_dataloader
        )

        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_accuracy": np.mean(valid_scores["accuracy"]),
                "valid_precision": np.mean(valid_scores["precision"]),
                "valid_recall": np.mean(valid_scores["recall"]),
                "valid_auroc": np.mean(valid_scores["auroc"]),
                "valid_f1": np.mean(valid_scores["f1"]),
                "lr": optimizer.param_groups[0]["lr"],
            }
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

            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{Config.save_path}/model_best.pth")
            print(f"MODEL SAVED!{Style.RESET_ALL}")

        torch.save(model.state_dict(), f"{Config.save_path}/model_last.pth")

    run.finish()
