<div align="center">
    <h1>
    TamMalKavacham
    </h1>
    <p>
      Abusive content detection against women and marginalized groups in Tamil and Malayalam languages.
    </p>
</div>

<div align="center">
  <img src="https://img.shields.io/pypi/pyversions/tammalkavacham?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-%20-orange?logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/github/license/Luxshan2000/tammalkavacham?logo=open-source-initiative&logoColor=white" alt="License" />
</div>

---

## 🛠️ Installation  

Install TamMalKavacham via [PyPI](https://pypi.org/project/tammalkavacham):  
```bash
pip install tammalkavacham
```  

---

## 💡 Quick Start  

### Load the Library  
```python
from tammalkavacham import AbuseDetector

# Initialize the detector
detector = AbuseDetector()
```

### Predict Abusive Content  
```python
text = "Example abusive text in Tamil or Malayalam"
result = detector.predict(text)

if result:
    print("⚠️ Abusive content detected!")
else:
    print("✔️ Text is clean.")
```  

---

## 🚀 Key Features  
- **Multilingual Detection**: Designed for Tamil 🇮🇳 and Malayalam 🇮🇳 text.  
- **Plug-and-Play**: No complex setup. Just install, load, and detect!  
- **Efficient Local Processing**: Downloads the pre-trained model on first use.  
- **Customizable**: Extendable for additional use cases with minor adjustments.  

---

## 📚 Documentation  

Full documentation is available at [https://yourusername.github.io/tammalkavacham](https://yourusername.github.io/tammalkavacham).  

### API Reference  
#### `AbuseDetector()`  
Initialize the detector instance.  

#### `.predict(text: str) -> bool`  
Predicts whether the input text is abusive.  
- **Parameters:**  
  - `text` (str): Input string in Tamil or Malayalam.  
- **Returns:**  
  - `bool`: `True` if abusive, `False` otherwise.  

---

## 🔧 Development  

### Clone the Repository  
```bash
git clone https://github.com/yourusername/tammalkavacham.git
cd tammalkavacham
```  

### Install Dependencies  
Use [Poetry](https://python-poetry.org/) to manage dependencies:  
```bash
poetry install
```
---

## 🌍 Supported Languages  

| Language     | Script          | Status       |  
|--------------|-----------------|--------------|  
| **Tamil**    | Tamil script    | ✅ Supported |  
| **Malayalam**| Malayalam script| ✅ Supported |  

---

## 🤝 Contribution  

Contributions are welcome! Follow the [Contribution Guidelines](CONTRIBUTING.md) for details.  

### Steps to Contribute  
1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m "Add feature-name"`.  
4. Push to the branch: `git push origin feature-name`.  
5. Open a pull request.  

---

## 📄 License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---

## 🙌 Acknowledgements  

Special thanks to the contributors and the open-source community for making this project possible!  

---

## 📧 Contact  

For questions or support, contact **Luxshan Thavarasa**:  
📧 Email: [luxshanlux2000@gmail.com](mailto:luxshanlux2000@gmail.com)  
🌐 LinkedIn: [linkedin.com/in/luxshan-thavarasa](https://www.linkedin.com/in/luxshan-thavarasa)  

---  

## ⭐ Support  

If you like this project, please consider giving it a ⭐ on [GitHub](https://github.com/yourusername/tammalkavacham)!  
