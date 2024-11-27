def load_model():
    from tensorflow.keras.models import load_model
    model = load_model('models/deepfake_audio_detection_model.h5')
    return model

def preprocess_audio(file_path):
    import librosa
    import numpy as np

    # Load the audio file
    audio, sr = librosa.load(file_path, sr=16000)

    # Normalize the audio
    audio = audio / np.max(np.abs(audio))

    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_resized = np.resize(mfccs, (128, 128))
    mfccs_3d = np.stack((mfccs_resized,)*3, axis=-1)

    return np.array([mfccs_3d])

def generate_spectrogram(file_path):
    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    import os

    audio, sr = librosa.load(file_path, sr=16000)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Ensure the static directory exists
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    spectrogram_path = os.path.join(static_dir, 'spectrogram.png')
    plt.savefig(spectrogram_path)
    plt.close()
    return spectrogram_path

def predict(model, features):
    prediction = model.predict(features)
    return prediction[0][0]  # Assuming binary classification (real/fake)