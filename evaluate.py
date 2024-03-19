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

humour2label = {'not_funny': 0, 'funny': 1, 'very_funny': 2, 'hilarious': 3}
sarcasm2label = {'not_sarcastic': 0, 'general': 1, 'twisted_meaning': 2, 'very_twisted': 3}
offensive2label = {'not_offensive': 0, 'slight': 1, 'very_offensive': 2, 'hateful_offensive': 3}
motivational2label = {'not_motivational': 0, 'motivational': 1}
overall_sentiment2label = {"very_positive": 0, "positive": 1, "neutral": 2, "negative": 3, "very_negative": 4}

label2humour = {0: 'not_funny', 1: 'funny', 2: 'very_funny', 3: 'hilarious'}
label2sarcasm = {0: 'not_sarcastic', 1: 'general', 2: 'twisted_meaning', 3: 'very_twisted'}
label2offensive = {0: 'not_offensive', 1: 'slight', 2: 'very_offensive', 3: 'hateful_offensive'}
label2motivational = {0: 'not_motivational', 1: 'motivational'}
label2overall_sentiment = {0: 'very_positive', 1: 'positive', 2: 'neutral', 3: 'negative', 4: 'very_negative'}

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
    humour = str(row['humour'].values[0]).lower()
    sarcasm = str(row['sarcasm'].values[0]).lower()
    offensive = str(row['offensive'].values[0]).lower()
    motivational = str(row['motivational'].values[0]).lower()
    overall_sentiment = str(row['overall_sentiment'].values[0]).lower()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    token = tokenizer(
        text=text,
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    label_humour = humour2label[humour]
    label_sarcasm = sarcasm2label[sarcasm]
    label_offensive = offensive2label[offensive]
    label_motivational = motivational2label[motivational]
    label_overall_sentiment = overall_sentiment2label[overall_sentiment]

    checkpoint = os.path.join(args.save_path, 'model_best.pth')

    emotion_names = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']
    emotions = [humour, sarcasm, offensive, motivational, overall_sentiment]

    with torch.no_grad():
        model = MemotionModel(task=1).cuda()
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        pred = model.forward(image=img,
                             input_ids=token['input_ids'].to(args.device).squeeze(),
                             attention_mask=token['attention_mask'].to(args.device).squeeze(), )
        print('--- Results using only image input ---')
        print('humour of the meme:', label2humour[int(torch.argmax(pred[0]))])
        print('sarcasm of the meme:', label2sarcasm[int(torch.argmax(pred[1]))])
        print('offensive of the meme:', label2offensive[int(torch.argmax(pred[2]))])
        print('motivational of the meme:', label2motivational[int(torch.argmax(pred[3]))])
        print('overall_sentiment of the meme:', label2overall_sentiment[int(torch.argmax(pred[4]))])

        model = MemotionModel(task=2).cuda()
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        pred = model.forward(image=img,
                             input_ids=token['input_ids'].to(args.device),
                             attention_mask=token['attention_mask'].to(args.device))
        print('--- Results using only text input ---')
        print('humour of the meme:', label2humour[int(torch.argmax(pred[0]))])
        print('sarcasm of the meme:', label2sarcasm[int(torch.argmax(pred[1]))])
        print('offensive of the meme:', label2offensive[int(torch.argmax(pred[2]))])
        print('motivational of the meme:', label2motivational[int(torch.argmax(pred[3]))])
        print('overall_sentiment of the meme:', label2overall_sentiment[int(torch.argmax(pred[4]))])

        model = MemotionModel(task=3).cuda()
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        pred = model.forward(image=img,
                             input_ids=token['input_ids'].to(args.device),
                             attention_mask=token['attention_mask'].to(args.device))
        print('--- Results using both image and text input ---')
        print('humour of the meme:', label2humour[int(torch.argmax(pred[0]))])
        print('sarcasm of the meme:', label2sarcasm[int(torch.argmax(pred[1]))])
        print('offensive of the meme:', label2offensive[int(torch.argmax(pred[2]))])
        print('motivational of the meme:', label2motivational[int(torch.argmax(pred[3]))])
        print('overall_sentiment of the meme:', label2overall_sentiment[int(torch.argmax(pred[4]))])

        print('--- Ground truth emotions ---')
        print('Ground truth humour:', humour)
        print('Ground truth sarcasm:', sarcasm)
        print('Ground truth offensive:', offensive)
        print('Ground truth motivational:', motivational)
        print('Ground truth overall_sentiment:', overall_sentiment)
