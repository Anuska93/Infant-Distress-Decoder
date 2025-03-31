import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# Paths
dataset_path = "C:/Users/Ankush/Desktop/Anuska/projects/BabyCry/donateacry_corpus"  # Update this path

# Define categories
categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'complaining']

# Feature extraction function
def extract_features(file_path, max_frames=100):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract 40 MFCCs

        # Pad or truncate
        if mfccs.shape[1] < max_frames:
            pad_width = max_frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_frames]

        return np.expand_dims(mfccs, axis=-1)  # Add channel dimension for CNN
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Prepare data
def prepare_data():
    features, labels = [], []
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(category)
    return np.array(features), np.array(labels)

# Load dataset
X, y = prepare_data()

# Print dataset distribution
print(f"Original dataset distribution: {Counter(y)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handle imbalanced data
X = X.reshape(X.shape[0], -1)  # Flatten for SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Reshape X back to CNN input format
X_resampled = X_resampled.reshape(X_resampled.shape[0], 40, X.shape[1] // 40, 1)

# Convert labels to one-hot encoding
y_resampled = to_categorical(y_resampled, num_classes=len(categories))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Input(shape=(40, X_train.shape[2], 1)),  # Explicit input shape
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to predict
def predict_audio(file_path):
    feature = extract_features(file_path)
    if feature is None:
        return "Error processing audio file"
    
    feature = np.expand_dims(feature, axis=0)  # Reshape for CNN
    prediction = model.predict(feature)
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

# Example usage
test_file = "C:/Users/Ankush/Desktop/Anuska/projects/BabyCry/donateacry_corpus/belly_pain/643D64AD-B711-469A-AF69-55C0D5D3E30F-1430138545-1.0-m-72-bp.wav"
print(f'In this audio the baby seems to be in: {predict_audio(test_file)}')
