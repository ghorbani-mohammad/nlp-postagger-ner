# Mohammad - NLP Project Portfolio

## Persian NLP Pipeline: POS Tagging & NER

### Project Overview
This project implements a comprehensive Persian text processing pipeline for Part-of-Speech (POS) tagging and Named Entity Recognition (NER) using machine learning approaches. The system processes Persian text with UTF-8 encoding and provides accurate linguistic analysis.

### Technical Implementation

#### POS Tagging System
- **Model**: Multilayer Perceptron (MLP) classifier from scikit-learn
- **Features**: Custom contextual feature engineering including:
  - Word position (first/last in sentence)
  - Previous and next word context
  - Morphological features (hyphenation detection, numeric identification)
  - Word-level characteristics
- **Data**: Persian text corpus with tab-separated word-tag pairs
- **Processing**: Sentence segmentation and tokenization
- **Output**: POS tags (N=noun, V=verb, ADJ=adjective, P=preposition, etc.)

#### Named Entity Recognition
- **Framework**: Stanford NLP toolkit integration
- **Model**: Pre-trained Stanford NER model for Persian text
- **Training**: Custom model training with Persian entity data
- **Evaluation**: Entity-level precision, recall, and F1-score metrics
- **Entity Types**: Person names, locations, organizations, and other named entities

#### Data Pipeline
- **Input**: Raw Persian text files (UTF-8 encoded)
- **Preprocessing**: Sentence boundary detection and word-tag separation
- **Feature Extraction**: Contextual feature generation for each word
- **Model Training**: MLP classifier training with feature vectors
- **Evaluation**: Confusion matrices and performance metrics
- **Output**: Tagged text with linguistic annotations

### Technologies Used
- **Python**: Core implementation language
- **NLTK**: Natural language processing toolkit
- **scikit-learn**: Machine learning pipeline and MLPClassifier
- **Stanford NLP**: Named entity recognition
- **pandas & numpy**: Data manipulation
- **joblib**: Model serialization
- **Jupyter Notebooks**: Development environment

### Project Files
- `Section1_POS.ipynb`: POS tagging implementation
- `Section2_NER.ipynb`: Named entity recognition implementation
- `Data/`: Persian text corpora for training and testing
- `NNModel.joblib`: Serialized trained model
- `trained_model.ser.gz`: Stanford NER model file

### Key Features
- Multilingual support for Persian text processing
- Custom feature engineering for contextual analysis
- Comprehensive evaluation metrics
- Model persistence and deployment capabilities
- Robust data preprocessing pipeline
