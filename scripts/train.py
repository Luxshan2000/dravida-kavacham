import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer
from model import AbusiveCommentClassifier
import torch.nn as nn
from scripts.utils import load_data


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_texts_ta, train_labels_ta = load_data('./data/train_ta.csv')
    train_texts_ma, train_labels_ma = load_data('./data/train_ma.csv')
    train_texts = train_texts_ta + train_texts_ma
    train_labels = train_labels_ta + train_labels_ma

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = AbusiveCommentClassifier()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        total_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Compute loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), './model/model.pth')
    print("Model saved to './model/model.pth'")

if __name__ == "__main__":
    train()
