# Skip-gram Model for Vietnamese Corpus

## Introduction
This repository contains the implementation and training of a **Skip-gram** model on a Vietnamese text corpus. The project was conducted as part of the **Natural Language Processing** course at Saigon University.

The Skip-gram model is a neural network designed to learn word embeddings, enabling the discovery of semantic relationships between words. This implementation focuses on Vietnamese, leveraging diverse text data to improve model generalization.

## Features
- Implements a Skip-gram model for word embedding.
- Preprocessing Vietnamese text with tools like [PyVi](https://pypi.org/project/pyvi/).
- Evaluates embeddings using cosine similarity for synonym and antonym word pairs.
- Includes experiments comparing models trained with different learning rates.

## Dataset
- The training data is sourced from a public repository on GitHub: [news-corpus](https://github.com/binhvq/news-corpus).
- Consists of over 14 million articles from various Vietnamese news outlets, ensuring diverse and realistic language representation.
- After preprocessing:
  - Number of sentences: 1000
  - Vocabulary size: 2651
  - Total word pairs: 23,460

## Preprocessing Steps
1. Convert text to lowercase for consistency.
2. Remove punctuation and special characters.
3. Tokenize using the **ViTokenizer** library.
4. Remove stopwords using a predefined list of Vietnamese stopwords.
5. Normalize data structure for model input.

## Model Details
- **Architecture**:
  - Input Layer: One-hot encoded word vectors.
  - Hidden Layer: Embedding layer to learn word representations.
  - Output Layer: Softmax for context word prediction.
- **Training Configuration**:
  - Embedding size: 100
  - Batch size: 1
  - Epochs: 100
  - Learning rates: 0.01 (Model V1), 0.05 (Model V2)
  - Loss function: Cross-Entropy

## Results
- Model V1 (learning rate 0.01) produced better embeddings, particularly for synonym pairs.
- Cosine similarity analysis showed higher semantic accuracy with Model V1 compared to Model V2.
- Example results:
  - Synonym pair "học" ↔ "giáo_dục": High similarity.
  - Antonym pair "tăng" ↔ "giảm": Clear semantic contrast.

## Challenges and Lessons Learned
- Handling noisy Vietnamese text from diverse sources.
- Selecting optimal hyperparameters for efficient training.
- Evaluating embeddings with limited computational resources.

## How to Use
### Prerequisites
- Python 3.8 or higher
- Required libraries: `pyvi`, `numpy`, `scipy`

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/skipgram-vietnamese.git
   cd skipgram-vietnamese
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```python
   python train.py
   ```
4. Evaluate embeddings:
   ```python
   python evaluate.py
   ```

## References
1. Mikolov, T., et al. (2013). [Distributed representations of words and phrases](https://arxiv.org/abs/1310.4546).
2. Rong, X. (2014). [Word2vec parameter learning explained](https://arxiv.org/abs/1411.2738).
3. [PyVi Documentation](https://pypi.org/project/pyvi/).
4. Binh, V. Q. (n.d.). [news-corpus on GitHub](https://github.com/binhvq/news-corpus).
5. [Vietnamese stopwords](https://github.com/stopwords/vietnamese-stopwords).

## Author
- **Dương Văn Sìnl**  
  Student at Saigon University, Faculty of Information Technology  
  Contact: tinbalon3@gmail.com

