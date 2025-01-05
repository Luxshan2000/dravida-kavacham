from pathlib import Path
import requests
from tammalkavacham.model import _AbusiveCommentClassifier
import torch
from transformers import XLMRobertaTokenizer
import gdown


MODEL_URL = "https://drive.google.com/uc?id=1RMi57aiqvQw79wlhL9IVFSqgIgBeZdMo"
MODEL_DIR = Path.home() / ".abusedetect_model"
MODEL_PATH = MODEL_DIR / "abusive_detector.pth"


class AbuseDetector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _AbusiveCommentClassifier().to(self.device)
        self._load_model()

    def _download_model(self):
        if not MODEL_PATH.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print("Downloading model...")
            gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
            print(f"Model downloaded to {MODEL_PATH}")
        else:
            print(f"Model already exists at {MODEL_PATH}")

    def _load_model(self):
        self._download_model()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

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