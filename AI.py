import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increase to 20 or more
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    features = np.concatenate([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0)
    ])
    return features


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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build a classifier (Multi-layer Perceptron)
grid_search = GridSearchCV(RandomForestClassifier(n_estimators= 100, class_weight='balanced',random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Step 4: Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)