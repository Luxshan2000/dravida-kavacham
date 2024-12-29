import torch
from sklearn.metrics import classification_report

def evaluate():
    model = EnhancedTextEncoder()
    model.load_state_dict(torch.load('./model/pretrained_model/enhanced_model.pth'))
    model.eval()

    dev_texts, dev_labels = ...  # Load from development.csv
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    dataset = TextDataset(dev_texts, dev_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
