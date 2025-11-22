## RNN IMDB Sentiment Model – Line by Line Explanation

This document mirrors the code from `rnn.ipynb` and adds an explanation for every line. The goal is to clarify what each statement does and why it is needed.

> Note: Indices 0–2 in the IMDB dataset are reserved (padding, start token, unknown). When decoding, we subtract 3 to map back to the original `word_index` entries.

---
### Cell 1 – Imports
```python
import numpy as np                      # Numerical operations (arrays, efficient math)
import tensorflow as tf                # Deep learning framework providing Keras high-level API
from tensorflow.keras.preprocessing import sequence  # Tools for sequence padding/utilities
from tensorflow.keras.models import Sequential       # Sequential container for stacking layers
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense  # Layer classes used in the model
from tensorflow.keras.datasets import imdb           # IMDB dataset loader (pre-tokenized reviews)
```

---
### Cell 2 – Load the Dataset
```python
max_features = 10000                    # Limit vocabulary size to the 10,000 most frequent words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  # Load tokenized reviews & labels

print("x_train shape:", x_train.shape)  # Show number of training samples and variable-length sequences
print("x_test shape:", x_test.shape)    # Show number of test samples and variable-length sequences
print("y_train shape:", y_train.shape)  # Show shape of training labels (binary sentiment)
print("y_test shape:", y_test.shape)    # Show shape of test labels
```

---
### Cell 3 – Inspect a Sample Review
```python
sample_review = x_train[0]              # Take first encoded review (list of integer token IDs)
sample_label = y_train[0]               # Corresponding sentiment label (0=negative, 1=positive)
print("Sample review (token indices):", sample_review)  # Display raw token sequence
print("Sample label (0=negative, 1=positive):", sample_label)  # Display label for context
```

---
### Cell 4 – Build Reverse Word Index
```python
word_index = imdb.get_word_index()      # Dict mapping words -> integer indices (global IMDB vocabulary)
word_index                             # (Notebook echo; in script would be removed – shows the dict)
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])  # Invert mapping: index -> word
reversed_word_index                     # (Notebook echo) Enables decoding integer sequences to words
```

---
### Cell 5 – Decode Sample Review
```python
decoded_review = ' '.join([reversed_word_index.get(i - 3, '?') for i in sample_review])  # Shift indices by 3 and map
print("Decoded review:", decoded_review)  # Human-readable approximation (unknown -> '?')
```
Explanation: Subtract 3 because IMDB reserves indices 0,1,2; dictionary starts at original words beyond those.

---
### Cell 6 – Pad Sequences
```python
from tensorflow.keras.preprocessing import sequence  # (Re-import; already imported earlier)

maxlen = 500                              # Target fixed length for each review (truncate/pad)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  # Left-pad shorter reviews with zeros
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)    # Same for test set
print("Padded x_train :", x_train)        # Display padded training data (2D array: samples x maxlen)
print("Padded x_test :", x_test)          # Display padded test data
```

---
### Cell 7 – Define Model Architecture
```python
model = Sequential()                      # Initialize an empty sequential model
model.add(Embedding(max_features, 128, input_length=maxlen))  # Embedding: maps token IDs -> 128-d vectors
# model.add(Embedding(max_features, 128, input_shape=(maxlen,)))  # Alternate form (commented for summary clarity)
model.add(SimpleRNN(128, activation='relu'))  # Recurrent layer with 128 units; processes sequence data
model.add(Dense(1, activation='sigmoid'))     # Output layer for binary classification (probability of positive)
model.summary()                               # Print model architecture details and parameter counts
```

---
### Cell 8 – Early Stopping Callback
```python
from tensorflow.keras.callbacks import EarlyStopping  # Utility to stop training when no improvement
early_stopping = EarlyStopping(monitor='val_loss',    # Watch validation loss metric
                               patience=5,            # Allow 5 epochs without improvement before stopping
                               restore_best_weights=True)  # Revert to weights from best epoch
early_stopping                                     # (Notebook echo) Confirms callback object
```

---
### Cell 9 – Compile and Train Model
```python
model.compile(loss='binary_crossentropy',            # Appropriate loss for binary classification
              optimizer='adam',                     # Adaptive learning rate optimizer
              metrics=['accuracy'])                 # Track accuracy during training/validation
history = model.fit(x_train, y_train,               # Training data and labels
                    batch_size=32,                  # Samples per gradient update
                    epochs=10,                      # Maximum number of full training passes
                    validation_split=0.2,           # Hold out 20% of training data for validation
                    callbacks=[early_stopping])     # Apply early stopping to prevent overfitting
history                                             # (Notebook echo) Training history object
```

---
### Cell 10 – Save Model
```python
model.save('rnn_imdb_model.h5')          # Serialize trained weights + architecture to disk
```

---
### Cell 11 – Reload Model for Inference
```python
from tensorflow.keras.models import load_model  # Import loader to reconstruct saved model
model = load_model('rnn_imdb_model.h5')         # Load the saved model from file
model.summary()                                 # Verify architecture matches expectations
```

---
### Cell 12 – Helper: Decode Review Function
```python
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])  # Convert IDs -> words
```
Purpose: Helps interpret encoded sequences and debug preprocessing/model decisions.

### Cell 12 (continued) – Preprocess User Input
```python
def preprocess_user_input(text):
    # Tokenize and encode the user input
    words = text.lower().split()                     # Simple whitespace tokenization + lowercase normalization
    encoded_review = [reversed_word_index.get(word, 2) + 3 for word in words]  # Map words -> indices (fallback=2)
    # Pad the encoded input to the maximum length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)      # Produce shape (1, 500)
    return padded_review
```
Note: Using `reversed_word_index.get(word, 2)` assumes words already in reverse mapping; typical approach would use `word_index`. This works only if tokens exist (else unknown index 2). Adding 3 maintains offset symmetry with dataset encoding.

---
### Cell 13 – Prediction Function
```python
def predict_sentiment(review_text):
    preprocessed_review = preprocess_user_input(review_text)          # Convert raw text -> padded token tensor
    prediction = model.predict(preprocessed_review)                   # Forward pass; output shape (1, 1)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"  # Threshold at 0.5 for class label
    return sentiment, prediction[0][0]                                # Return label and raw probability
```

---
### Cell 14 – Test Prediction
```python
user_review = "The movie was fantastic! I really loved it and would watch it again."  # Sample input text
sentiment, confidence = predict_sentiment(user_review)   # Run inference
print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")  # Display result with formatted confidence
```

---
### Cell 15 – Inspect Encoding & Decoding of User Review
```python
encoded_full = preprocess_user_input(user_review)[0]      # Extract padded sequence array (length 500)
encoded_nopad = [i for i in encoded_full if i != 0]       # Remove leading padding zeros for cleaner decode

print("Encoded:", encoded_full[-20:])                     # Show last 20 token IDs (tail portion)
print("Decoded:", decode_review(encoded_nopad))           # Reconstruct words from token IDs

# (Optional) You can compare to the original to see unknown words mapped to '?'.
print("\nOriginal review:")
print(user_review)                                        # Show original raw text for comparison
```

---
## Key Concepts Recap
- **Tokenization**: Reviews are already integer-encoded by dataset loader (frequency-based index).
- **Padding**: Ensures uniform length for batch processing by RNN (shape consistency).
- **Embedding Layer**: Learns dense vector representations of tokens during training.
- **SimpleRNN**: Processes sequence left-to-right; hidden state summarizes context.
- **Activation Choices**: `relu` in RNN (may risk exploding states vs. `tanh` but used here experimentally); `sigmoid` for final probability.
- **EarlyStopping**: Prevents overfitting by monitoring validation loss.
- **Decoding**: Aids interpretability and debugging by mapping indices back to words.
- **Thresholding**: Simple rule (>= 0.5) for binary sentiment decision.

## Potential Improvements
- Use `Tokenizer` from Keras for consistent preprocessing and OOV handling.
- Switch `SimpleRNN` to `LSTM` or `GRU` for better long-term dependency capture.
- Add dropout to reduce overfitting.
- Normalize punctuation and handle contractions for better token matching.
- Store and reuse a fitted tokenizer instead of relying on global `word_index` mapping.

---
If you want, I can also produce a version with refactored preprocessing using `Tokenizer`. Let me know.
