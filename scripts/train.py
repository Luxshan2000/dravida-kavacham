import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer
from model import AbusiveCommentClassifier
import torch.nn as nn
from scripts.utils import load_data

print("Imported!!")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train():
    train_texts_ta, train_labels_ta = load_data('./data/train_ta.csv')
    train_texts_ma, train_labels_ma = load_data('./data/train_ta.csv')
    train_texts = train_texts_ta + train_texts_ma
    train_labels = train_labels_ta + train_labels_ma

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


    model = AbusiveCommentClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _ in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(outputs, batch['labels'])
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './model/model.pth')

if __name__ == "__main__":
    train()
