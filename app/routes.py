from flask import Blueprint, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
from app.utils import preprocess_audio, load_model, predict, generate_spectrogram

app_bp = Blueprint('app_bp', __name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

def create_app():
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(app_bp)
    return app

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audioFile' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['audioFile']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            audio_features = preprocess_audio(file_path)
            model = load_model()
            result = predict(model, audio_features)
            classification = 'Real' if result > 0.5 else 'Fake'
            spectrogram_path = generate_spectrogram(file_path)
            return render_template('index.html', result=f'The audio is classified as: {classification}', score=f'{result:.2f}', spectrogram=url_for('static', filename='spectrogram.png'))
        else:
            return render_template('index.html', message='Invalid file format')
    return render_template('index.html')