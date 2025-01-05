<div align="center">
    <h1>
    kavacham
    </h1>
    <p>
        kavacham is an open-source tool for detecting abusive content in Dravidian focused on harmful language targeting women.</p>
</div>

<div align="center">
  <img src="https://img.shields.io/pypi/pyversions/kavacham?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.0%2B-red?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/github/license/Luxshan2000/kavacham" alt="MIT License" />
</div>

---

## 🛠️ Installation  

Install kavacham via [PyPI](https://pypi.org/project/kavacham):  
```bash
pip install kavacham
```  

---

## 💡 Quick Start  

### Load the Library  
```python
from kavacham import AbuseDetector

# Initialize the detector
detector = AbuseDetector()
```

### Predict Abusive Content  
```python
text = "Example abusive text in Tamil or Malayalam"
result = detector.predict(text)

if result == "Abusive":
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

Full documentation is available at [Home](https://yourusername.github.io/tammalkavacham).  

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

## 📧 Contact  

For questions or support, contact **Luxshan Thavarasa**:  
📧 Email: [luxshanlux2000@gmail.com](mailto:luxshanlux2000@gmail.com)  
🌐 LinkedIn: [linkedin.com/in/luxshan-thavarasa](https://www.linkedin.com/in/luxshan-thavarasa)  

---  

## ⭐ Support  

If you like this project, please consider giving it a ⭐ on [kavacham](https://github.com/Luxshan2000/kavacham)!
