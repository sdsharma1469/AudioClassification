import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def process_audio_files(directory):
    features_list = []
    labels_list = []
    
    # Use os.walk() to traverse through all subdirectories
    for subdir, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.ogg'):
                file_path = os.path.join(subdir, file_name)
                features = extract_features(file_path)
                label = os.path.basename(subdir)  # Use the subdirectory name as the label
                features_list.append(features)
                labels_list.append(label)
    
    return np.array(features_list), np.array(labels_list)

# Example usage
directory = 'test'  # Set the root directory to start from
features, labels = process_audio_files(directory)

print("Extracted Features Shape:", features.shape)
print("Labels:", labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 2: Build a classifier (Multi-layer Perceptron)
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42)

# Step 3: Train the model
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)