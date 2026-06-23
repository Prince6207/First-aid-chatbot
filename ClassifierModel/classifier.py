import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

DATA_FILE = 'intentDataset.csv'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
CONFIDENCE_THRESHOLD = 0.50 # SVM probabilities scale differently than Neural Nets

print("Loading data and embedding model (this may take a moment to download the first time)...")
df = pd.read_csv(DATA_FILE)

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Encoding sentences to vectors...")
X = embedder.encode(df['patterns'].tolist(), show_progress_bar=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['tag'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Support Vector Machine (SVM)...")
svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print("\nEvaluating Model on Test Set:")
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def predict_intent(sentence):
    """
    Predicts intent using Transformer Embeddings + SVM.
    Applies a confidence threshold to catch out-of-scope ("None") queries.
    """
    vector = embedder.encode([sentence])
    
    probabilities = svm_model.predict_proba(vector)[0]
    max_prob_index = np.argmax(probabilities)
    max_prob_value = probabilities[max_prob_index]
    
    if max_prob_value < CONFIDENCE_THRESHOLD:
        return "None", max_prob_value
        
    predicted_tag = label_encoder.inverse_transform([max_prob_index])[0]
    return predicted_tag, max_prob_value

if __name__ == "__main__":
    print("\n--- Testing Predictions ---")
    test_sentences = [
        "What causes anemia?",                # Expected: cause
        "What are the signs of low iron?",    # Expected: symptom
        "How can I prevent it?",              # Expected: prevention
        "Can I eat spinach to fix it?",       # Expected: food
        "What is the weather like today?",    # Expected: None
        "Can you play some music?"            # Expected: None
    ]
    
    for text in test_sentences:
        tag, confidence = predict_intent(text)
        print(f"Input: '{text}'")
        print(f"Predicted Tag: {tag} (Confidence: {confidence:.2f})\n")

    with open('svm_intent_model.pkl', 'wb') as f: pickle.dump(svm_model, f)
    with open('label_encoder.pkl', 'wb') as f: pickle.dump(label_encoder, f)