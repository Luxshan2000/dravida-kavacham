import torch
import pandas as pd

def infer():
    model = EnhancedTextEncoder()
    model.load_state_dict(torch.load('./model/pretrained_model/enhanced_model.pth'))
    model.eval()

    submission_texts = ...  # Load from submission.csv
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    predictions = []
    for text in submission_texts:
        inputs = tokenizer(
            text, max_length=128, padding='max_length', truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            pred = torch.argmax(outputs, dim=1).item()
            predictions.append(pred)

    submission = pd.DataFrame({'text': submission_texts, 'label': predictions})
    submission.to_csv('./output/predictions/submission_with_predictions.csv', index=False)

if __name__ == "__main__":
    infer()
