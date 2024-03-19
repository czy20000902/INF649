import argparse
import os
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple

import albumentations as A
import logging
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from colorama import Back, Fore, Style
from torch import nn, optim
from transformers import CLIPTokenizer

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
logging.basicConfig(filename='evaluate.log', level=logging.INFO)

parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--task', type=int, default=1)
parser.add_argument('--root', type=str, default='/users/eleves-a/2022/zhaoyang.chen/memotion_dataset_7k/')
# parser.add_argument('--root', type=str, default='E:/Dataset/memotion_dataset_7k/')
parser.add_argument('--img_name', type=str, default='image_1.jpg')
parser.add_argument('--save_path', type=str, default='checkpoints')
parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224])

# Config
parser.add_argument('--num_classes', type=int, default=5)

# Device
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

emotion2label = {"very_positive": 0,
                 "positive": 1,
                 "neutral": 2,
                 "negative": 3,
                 "very_negative": 4}

label2emotion = {0: 'very_positive',
                 1: 'positive',
                 2: 'neutral',
                 3: 'negative',
                 4: 'very_negative'}

if __name__ == "__main__":
    print('#' * 18)
    print(f'### Evaluating ###')
    print('#' * 18)

    img_transforms = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1]),
        ToTensorV2(),
    ])

    image_path = os.path.join(args.root, 'images', args.img_name)
    label_path = os.path.join(args.root, 'labels.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(label_path)
    row = df[df['image_name'] == args.img_name]

    img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
    img = img_transforms(image=img)['image']
    img = img.to(args.device).unsqueeze(0)

    # Text
    text = str(row['text_corrected']).lower()
    emotion = str(row['overall_sentiment'].values[0]).lower()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    token = tokenizer(
        text=text,
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    label = emotion2label[emotion]

    checkpoint = os.path.join(args.save_path, 'model_best_0.pth')

    with torch.no_grad():
        model_1 = MemotionModel(task=1, numclasses=args.num_classes).cuda()
        model_1.load_state_dict(torch.load(checkpoint))
        model_1.eval()
        pred_1 = model_1.forward(image=img,
                                 input_ids=token['input_ids'].to(args.device).squeeze(),
                                 attention_mask=token['attention_mask'].to(args.device).squeeze(),
                                 label=torch.LongTensor([label]).to(args.device).squeeze())
        print('Emotion of the meme (using only image input):', label2emotion[int(torch.argmax(pred_1))])

        model_2 = MemotionModel(task=2, numclasses=args.num_classes).cuda()
        model_2.load_state_dict(torch.load(checkpoint))
        model_2.eval()
        pred_2 = model_2.forward(image=img,
                                 input_ids=token['input_ids'].to(args.device),
                                 attention_mask=token['attention_mask'].to(args.device),
                                 label=torch.LongTensor([label]).to(args.device).squeeze())
        print('Emotion of the meme (using only text input):', label2emotion[int(torch.argmax(pred_2))])

        model = MemotionModel(task=3, numclasses=args.num_classes).cuda()
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        pred = model.forward(image=img,
                             input_ids=token['input_ids'].to(args.device),
                             attention_mask=token['attention_mask'].to(args.device),
                             label=torch.LongTensor([label]).to(args.device).squeeze())
        print('Emotion of the meme (using both image and text input):', label2emotion[int(torch.argmax(pred))])
        print('Ground truth emotion:', emotion)
