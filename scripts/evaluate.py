import torch
from sklearn.metrics import classification_report
from model import AbusiveCommentClassifier
from transformers import XLMRobertaTokenizer
from scripts.train import TextDataset
from torch.utils.data import DataLoader
from scripts.utils import load_data

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AbusiveCommentClassifier()
    model.to(device)
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()


    dev_texts_ta, dev_labels_ta = load_data('./data/dev_ta.csv')
    dev_texts_ma, dev_labels_ma = load_data('./data/dev_ma.csv')
    dev_texts = dev_texts_ta + dev_texts_ma
    dev_labels = dev_labels_ta + dev_labels_ma

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    dataset = TextDataset(dev_texts, dev_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
