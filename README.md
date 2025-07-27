# Persian NLP Pipeline: POS Tagging & Named Entity Recognition

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Persian Natural Language Processing pipeline that implements Part-of-Speech (POS) tagging and Named Entity Recognition (NER) using machine learning approaches. This project processes Persian text with UTF-8 encoding and provides accurate linguistic analysis for Persian language applications.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### POS Tagging System
- **Advanced MLP Classifier**: Multilayer Perceptron neural network for accurate POS prediction
- **Contextual Feature Engineering**: 
  - Word position analysis (first/last in sentence)
  - Previous and next word context
  - Morphological features (hyphenation, numeric detection)
  - Word-level characteristics
- **Persian Language Support**: Optimized for Persian text processing
- **High Accuracy**: Robust performance on Persian text corpora

### Named Entity Recognition
- **Stanford NLP Integration**: Leverages Stanford NLP toolkit for NER
- **Custom Persian Model**: Pre-trained model specifically for Persian entities
- **Entity Types**: Person names, locations, organizations, and other named entities
- **Comprehensive Evaluation**: Entity-level precision, recall, and F1-score metrics

### Data Pipeline
- **UTF-8 Encoding**: Full support for Persian Unicode characters
- **Automated Preprocessing**: Sentence boundary detection and word-tag separation
- **Feature Extraction**: Advanced contextual feature generation
- **Model Persistence**: Save and load trained models for deployment

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Java Runtime Environment (JRE) 1.8+ (for Stanford NER)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nlp-postagger-ner.git
   cd nlp-postagger-ner
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   ```

4. **Set up Java environment** (for Stanford NER)
   - Ensure Java is installed and `JAVA_HOME` is set
   - The project includes `stanford-ner.jar` and `trained_model.ser.gz`

## 📁 Project Structure

```
nlp-postagger-ner/
├── Data/                          # Persian text corpora
│   ├── POStrutf.txt              # POS training data (UTF-8)
│   ├── POSteutf.txt              # POS test data (UTF-8)
│   ├── NERtr.txt                 # NER training data
│   ├── NERte.txt                 # NER test data
│   ├── in.txt                    # Sample input
│   └── out.txt                   # Sample output
├── Section1_POS.ipynb            # POS tagging implementation
├── Section2_NER.ipynb            # NER implementation
├── NNModel.joblib                # Serialized POS model
├── trained_model.ser.gz          # Stanford NER model
├── stanford-ner.jar              # Stanford NER JAR file
├── Report.pdf                    # Detailed project report
├── SNLP_HW3.pdf                  # Assignment specification
└── README.md                     # This file
```

## 💻 Usage

### POS Tagging

1. **Open the POS notebook**
   ```bash
   jupyter notebook Section1_POS.ipynb
   ```

2. **Run the cells sequentially** to:
   - Load and preprocess Persian text data
   - Extract contextual features
   - Train the MLP classifier
   - Evaluate model performance
   - Save the trained model

### Named Entity Recognition

1. **Open the NER notebook**
   ```bash
   jupyter notebook Section2_NER.ipynb
   ```

2. **Execute the cells** to:
   - Load the pre-trained Stanford NER model
   - Process test data
   - Perform entity recognition
   - Calculate evaluation metrics

### Using Pre-trained Models

```python
# Load POS model
from joblib import load
pos_model = load('NNModel.joblib')

# Load NER model
from nltk.tag.stanford import StanfordNERTagger
ner_tagger = StanfordNERTagger('trained_model.ser.gz', 'stanford-ner.jar', encoding='utf8')

# Process Persian text
text = "متن فارسی برای پردازش"
# Apply POS tagging and NER
```

## 🔧 Technical Details

### Technologies Used
- **Python 3.7+**: Core implementation language
- **scikit-learn**: Machine learning pipeline and MLPClassifier
- **NLTK**: Natural language processing toolkit
- **Stanford NLP**: Named entity recognition framework
- **pandas & numpy**: Data manipulation and numerical operations
- **joblib**: Model serialization and persistence
- **Jupyter Notebooks**: Interactive development environment

### Feature Engineering (POS Tagging)
The POS tagging system uses sophisticated feature extraction:

```python
def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
    }
```

### Data Format
- **Training Data**: Tab-separated word-tag pairs
- **Encoding**: UTF-8 for Persian character support
- **Sentence Boundaries**: Marked with special tokens
- **Entity Labels**: BIO tagging scheme for NER

## 📊 Performance

### POS Tagging Metrics
- **Accuracy**: High performance on Persian text
- **Confusion Matrix**: Detailed analysis of tag predictions
- **Cross-validation**: Robust model evaluation

### NER Performance
- **Precision**: Entity-level precision metrics
- **Recall**: Comprehensive entity detection
- **F1-Score**: Balanced performance measure
- **Entity Types**: Support for multiple entity categories

## 🤝 Contributing

We welcome contributions to improve the Persian NLP pipeline! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Improvement
- Additional feature engineering for POS tagging
- Support for more entity types in NER
- Performance optimization
- Documentation enhancements
- Test coverage expansion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Stanford NLP Documentation](https://nlp.stanford.edu/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Persian Language Processing Resources](https://github.com/sobhe/hazm)

## 📞 Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the project documentation

---

**Note**: This project is designed specifically for Persian language processing and may require adjustments for other languages or specific use cases.
