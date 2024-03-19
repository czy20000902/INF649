import albumentations as A
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast, BertTokenizer


class MemotionDatasetSentiment(Dataset):
    def __init__(self, split=None, root='E:/Dataset/memotion_dataset_7k/', transform=None) -> None:
        super().__init__()
        # self.df = df
        self.split = split
        self.root = root
        self.df = pd.read_csv(os.path.join(root, 'labels.csv'))
        # if self.split == 'train':
        #     self.df = self.df.head(int(len(self.df) * 0.9))
        # else:
        #     self.df = self.df.tail(int(len(self.df) * 0.1))
        # self.df = self.df.rename(columns={0: "", 1: "image_name", 3: "text"})
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.transforms = transform
        self.emotion2label = {"very_positive": 0,
                              "positive": 1,
                              "neutral": 2,
                              "negative": 3,
                              "very_negative": 4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        image_path = os.path.join(self.root, 'images', row['image_name'])
        img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
        img = self.transforms(image=img)['image']

        # Text
        text = str(row['text_corrected']).lower()
        out = self.tokenizer(
            text=text,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # __import__('pprint').pprint(out)
        label = self.emotion2label[row['overall_sentiment']]

        return {
            'image': img,
            'input_ids': out['input_ids'].squeeze(),
            'attention_mask': out['attention_mask'].squeeze(),
            'label': torch.LongTensor([label]).squeeze()
        }


class MemotionDataset(Dataset):
    def __init__(self, split=None, root='E:/Dataset/memotion_dataset_7k/', transform=None) -> None:
        super().__init__()
        # self.df = df
        self.split = split
        self.root = root
        self.df = pd.read_csv(os.path.join(root, 'labels.csv'))
        if self.split != 'train':
            self.df = self.df.head(int(len(self.df) * 0.1))
        # if self.split == 'train':
        #     self.df = self.df.head(int(len(self.df) * 0.9))
        # else:
        #     self.df = self.df.tail(int(len(self.df) * 0.1))
        # self.df = self.df.rename(columns={0: "", 1: "image_name", 3: "text"})
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.transforms = transform
        cat_replace = {'not_funny': 0, 'funny': 1, 'very_funny': 2, 'hilarious': 3}
        self.df['humour'] = self.df['humour'].replace(cat_replace)
        cat_replace = {'not_sarcastic': 0, 'general': 1, 'twisted_meaning': 2, 'very_twisted': 3}
        self.df['sarcasm'] = self.df['sarcasm'].replace(cat_replace)
        cat_replace = {'not_offensive': 0, 'slight': 1, 'very_offensive': 2, 'hateful_offensive': 3}
        self.df['offensive'] = self.df['offensive'].replace(cat_replace)
        cat_replace = {'not_motivational': 0, 'motivational': 1}
        self.df['motivational'] = self.df['motivational'].replace(cat_replace)
        cat_replace = {"very_positive": 0, "positive": 1, "neutral": 2, "negative": 3, "very_negative": 4}
        self.df['overall_sentiment'] = self.df['overall_sentiment'].replace(cat_replace)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        image_path = os.path.join(self.root, 'images', row['image_name'])
        img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
        img = self.transforms(image=img)['image']

        # Text
        text = str(row['text_corrected']).lower()
        out = self.tokenizer(
            text=text,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # __import__('pprint').pprint(out)
        label_humour = row['humour']
        label_sarcasm = row['sarcasm']
        label_offensive = row['offensive']
        label_motivational = row['motivational']
        label_overall_sentiment = row['overall_sentiment']

        return {
            'image': img,
            'input_ids': out['input_ids'].squeeze(),
            'attention_mask': out['attention_mask'].squeeze(),
            'label_humour': torch.LongTensor([label_humour]).squeeze(),
            'label_sarcasm': torch.LongTensor([label_sarcasm]).squeeze(),
            'label_offensive': torch.LongTensor([label_offensive]).squeeze(),
            'label_motivational': torch.LongTensor([label_motivational]).squeeze(),
            'label_overall_sentiment': torch.LongTensor([label_overall_sentiment]).squeeze(),
        }
