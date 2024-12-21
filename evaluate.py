# === evaluate.py ===
import numpy as np
import matplotlib.pyplot as plt
import pickle

# === 1. HELPER FUNCTIONS ===
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model_data["embedding_matrix"], model_data["word_to_index"], model_data["index_to_word"], model_data["loss_history"]

# === 2. EVALUATION ===
if __name__ == '__main__':
    # Load models
    model_v1_path = "skipgram_model_v1.pkl"
    embedding_matrix_v1, word_to_index_v1, index_to_word_v1, loss_history_v1 = load_model(model_v1_path)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history_v1) + 1), loss_history_v1, label="Model v1")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate word similarity
    word_pairs = [
        ("học", "giáo_dục"),
        ("phạm_luật", "vi_phạm"),
        ("xử_lý", "giải_quyết"),
        ("kiểm_tra", "xác_minh"),
        ("học", "vi_phạm"),
        ("tăng", "giảm"),
        ("đất", "nước"),
        ("đồng_ý", "phản_đối")
    ]

    def calculate_similarity(embedding_matrix, word_to_index, word_pairs):
        similarities = []
        for word1, word2 in word_pairs:
            vec1 = embedding_matrix[word_to_index[word1]] if word1 in word_to_index else None
            vec2 = embedding_matrix[word_to_index[word2]] if word2 in word_to_index else None
            if vec1 is not None and vec2 is not None:
                similarity = cosine_similarity(vec1, vec2)
                similarities.append((word1, word2, similarity))
            else:
                similarities.append((word1, word2, None))
        return similarities

    similarities_v1 = calculate_similarity(embedding_matrix_v1, word_to_index_v1, word_pairs)

    print("Cosine Similarity - Model v1:")
    for word1, word2, sim in similarities_v1:
        print(f"{word1} ↔ {word2}: {sim if sim is not None else 'Not found in vocabulary'}")

    # Plot similarity scores
    words = [f"{word1} ↔ {word2}" for word1, word2, _ in similarities_v1]
    scores_v1 = [sim if sim is not None else 0 for _, _, sim in similarities_v1]

    plt.figure(figsize=(10, 6))
    plt.barh(words, scores_v1)
    plt.xlabel("Cosine Similarity")
    plt.title("Word Pair Similarity - Model v1")
    plt.tight_layout()
    plt.show()
