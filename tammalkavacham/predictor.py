from pathlib import Path
import requests
from tammalkavacham.model import _AbusiveCommentClassifier
import torch
from transformers import XLMRobertaTokenizer

MODEL_URL = "https://raw.githubusercontent.com/Luxshan2000/TamMalKavacham/main/model/abusive_detector.pth"
MODEL_DIR = Path.home() / ".abusedetect_model"
MODEL_PATH = MODEL_DIR / "abusive_detector.pth"


class AbuseDetector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _AbusiveCommentClassifier().to(self.device)
        self._load_model()

    def _download_model(self):
        if not MODEL_PATH.exists():
            print("Downloading model...")
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print(f"Model downloaded to {MODEL_PATH}")
        else:
            print(f"Model already exists at {MODEL_PATH}")

    def _load_model(self):
        self._download_model()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval(0)

    def predict(self,text):
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1).item()
            pred = "Abusive" if pred == 1 else "Non-Abusive"

        return pred