# Python
import os
import numpy as np
import librosa

def load_data(data_dir):
    X, y = [], []
    for label in ['real', 'fake']:
        class_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            X.append(spectrogram)
            y.append(0 if label == 'real' else 1)
    return np.array(X), np.array(y)