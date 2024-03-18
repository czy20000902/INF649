import albumentations as A
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, CLIPTokenizer, CLIPTokenizerFast, BertTokenizer

class MemotionDataset(Dataset):
    def __init__(self, split=None, root='E:/Dataset/memotion_dataset_7k/labels.csv', transform=None) -> None:
        super().__init__()
        # self.df = df
        self.df = pd.read_csv(root)
        # self.df = self.df.rename(columns={0: "", 1: "image_name", 3: "text"})
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.transforms = transform
        self.emotion2label = {"very_positive":0,
                              "positive":1,
                              "neutral": 2,
                              "negative": 3,
                              "very_negative": 4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        image_path = os.path.join('E:/Dataset/memotion_dataset_7k/images', row['image_name'])
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
