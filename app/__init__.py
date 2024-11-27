from flask import Flask

def create_app():
    app = Flask(__name__)
    # Configure your app here
    from app.routes import app_bp
    app.register_blueprint(app_bp)
    return app

# Additional configuration can be added here if needed

from app import routes