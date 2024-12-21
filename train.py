# === train.py ===
import numpy as np
import re
import string
from pyvi import ViTokenizer
from tqdm import tqdm
import pickle

# === 1. DATA PREPROCESSING ===
def preprocess_text(corpus):
    sentences = corpus.split('\n')
    processed_sentences = []
    with open("stopword/stopwords-vi.txt", "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
    for sentence in sentences:
        if sentence.strip():
            cleaned_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            tokens = ViTokenizer.tokenize(cleaned_sentence).split()
            filtered_tokens = [word for word in tokens if word not in stop_words]
            processed_sentences.append(filtered_tokens)
    return processed_sentences

def build_vocab(sentences):
    vocab = {}
    reverse_vocab = {}
    idx = 0
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = idx
                reverse_vocab[idx] = word
                idx += 1
    return vocab, reverse_vocab

def generate_training_data(sentences, window_size, vocabulary):
    training_data = []
    for sentence in sentences:
        for idx, word in enumerate(sentence):
            for neighbor in range(-window_size, window_size + 1):
                if neighbor == 0 or idx + neighbor < 0 or idx + neighbor >= len(sentence):
                    continue
                context_word = sentence[idx + neighbor]
                if word in vocabulary and context_word in vocabulary:
                    training_data.append((vocabulary[word], vocabulary[context_word]))
    return training_data

# === 2. SKIP-GRAM MODEL ===
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.rand(vocab_size, embedding_dim)
        self.W2 = np.random.rand(embedding_dim, vocab_size)

    def forward(self, center_word_idx):
        h = self.W1[center_word_idx]
        u = np.dot(h, self.W2)
        y_pred = softmax(u)
        return y_pred, h, u

    def backward(self, error, h, center_word_idx, learning_rate):
        dW2 = np.outer(h, error)
        dW1 = np.dot(self.W2, error)
        self.W1[center_word_idx] -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    def train(self, training_data, epochs, learning_rate):
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for center_word_idx, context_word_idx in tqdm(training_data):
                y_pred, h, u = self.forward(center_word_idx)
                error = y_pred.copy()
                error[context_word_idx] -= 1
                self.backward(error, h, center_word_idx, learning_rate)
                total_loss += -np.log(y_pred[context_word_idx])
            loss_history.append(total_loss / len(training_data))
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")
        return loss_history

# === 3. TRAINING ===
if __name__ == '__main__':
    with open('dataset/demo-title.txt', 'r', encoding='utf-8') as file:
        text_corpus = file.read()

    sentences = preprocess_text(text_corpus)
    vocabulary, reverse_vocab = build_vocab(sentences)
    training_data = generate_training_data(sentences, window_size=2, vocabulary=vocabulary)

    embedding_dim = 100
    skip_gram = SkipGramModel(len(vocabulary), embedding_dim)
    loss_history = skip_gram.train(training_data, epochs=100, learning_rate=0.01)

    def save_model(embedding_matrix, word_to_index, index_to_word, loss_history, file_path):
        model_data = {
            "embedding_matrix": embedding_matrix,
            "word_to_index": word_to_index,
            "index_to_word": index_to_word,
            "loss_history": loss_history
        }
        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_path}")

    save_model(skip_gram.W1, vocabulary, reverse_vocab, loss_history, "skipgram_model_v1.pkl")
