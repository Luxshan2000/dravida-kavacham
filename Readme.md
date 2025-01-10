<div align="center">
    <h1>
    DravidaKavacham
    </h1>
    <p>
        DravidaKavacham  is an open-source tool for detecting abusive content in Dravidian focused on harmful language targeting women.</p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.0%2B-red?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/github/license/Luxshan2000/dravida-kavacham" alt="MIT License" />
<img src="https://img.shields.io/pypi/dm/dravida-kavacham" alt="Downloads" />
</div>

---

## 🛠️ Installation  

Install DravidaKavacham via [PyPI](https://pypi.org/project/dravida-kavacham)
```bash
pip install dravida-kavacham
```  

---

## 💡 Quick Start  

### Load the Library  
```python
from dravida_kavacham import AbuseDetector

# Initialize the detector
detector = AbuseDetector()
```

### Predict Abusive Content  
```python
text = "மிகவும் அழகாக உள்ளது."
result = detector.predict(text)

if result == "Abusive":
    print("⚠️ Abusive content detected!")
else:
    print("✔️ Text is clean.")
```  

Example notebook on Kaggle: [visit](https://www.kaggle.com/code/luxluxshan/dravida-kavacham)

---

## 🚀 Key Features  
- **Multilingual Detection**: Designed for Tamil 🇮🇳 and Malayalam 🇮🇳 text.  
- **Plug-and-Play**: No complex setup. Just install, load, and detect!  
- **Efficient Local Processing**: Downloads the pre-trained model on first use.  
- **Customizable**: Extendable for additional use cases with minor adjustments.  

---

## 🌍 Supported Languages  

| Language     | Script          | Status       |  
|--------------|-----------------|--------------|  
| **Tamil**    | Tamil script    | ✅ Supported |  
| **Malayalam**| Malayalam script| ✅ Supported |  

---

## 📄 License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---

## 🙌 Acknowledgements  

Special thanks to the dataset authors and owners for providing the valuable resources that made this project possible!

---

## ⭐ Support  

If you like this project, please consider giving it a ⭐ on [DravidaKavacham](https://github.com/Luxshan2000/dravida-kavacham)!
