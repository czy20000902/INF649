import torch
from torch import nn
from transformers import CLIPTextModel
from model.mobileone import MobileOne


class MemotionModel(nn.Module):
    def __init__(self,
                 task=3,
                 numclasses_humour=4,
                 numclasses_sarcasm=4,
                 numclasses_offensive=4,
                 numclasses_motivational=2,
                 numclasses_overall_sentiment=5) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task

        self.numclasses_humour = numclasses_humour
        self.numclasses_sarcasm = numclasses_sarcasm
        self.numclasses_offensive = numclasses_offensive
        self.numclasses_motivational = numclasses_motivational
        self.numclasses_overall_sentiment = numclasses_overall_sentiment

        self.image_encoder = MobileOne(width_multipliers=[1, 1, 1, 1]).to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.alpha_img = torch.randn(size=(1,), requires_grad=True, device=self.device)
        self.alpha_txt = torch.randn(size=(1,), requires_grad=True, device=self.device)
        # self.fc = nn.Linear(768, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 5)
        self.linear_humour = nn.Linear(512, self.numclasses_humour)
        self.linear_sarcasm = nn.Linear(512, self.numclasses_sarcasm)
        self.linear_offensive = nn.Linear(512, self.numclasses_offensive)
        self.linear_motivational = nn.Linear(512, self.numclasses_motivational)
        self.linear_overall_sentiment = nn.Linear(512, self.numclasses_overall_sentiment)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, image, input_ids, attention_mask, *args, **kwargs):
        if self.task == 1:
            img_out = self.image_encoder.forward(image)
            wt_emb = self.alpha_img * img_out
            return self.linear_humour(wt_emb), self.linear_sarcasm(wt_emb), self.linear_offensive(
                wt_emb), self.linear_motivational(wt_emb), self.linear_overall_sentiment(wt_emb)
        if self.task == 2:
            txt_out = self.text_encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
            wt_emb = self.alpha_txt * txt_out.pooler_output
            return self.linear_humour(wt_emb), self.linear_sarcasm(wt_emb), self.linear_offensive(
                wt_emb), self.linear_motivational(wt_emb), self.linear_overall_sentiment(wt_emb)
        if self.task == 3:
            img_out = self.image_encoder.forward(image)
            txt_out = self.text_encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
            wt_emb = self.alpha_txt * txt_out.pooler_output + self.alpha_img * img_out
            return self.linear_humour(wt_emb), self.linear_sarcasm(wt_emb), self.linear_offensive(
                wt_emb), self.linear_motivational(wt_emb), self.linear_overall_sentiment(wt_emb)
