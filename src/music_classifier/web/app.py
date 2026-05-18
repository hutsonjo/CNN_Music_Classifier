"""Flask application factory and backend configuration."""

import os
from flask import Flask
from flask_cors import CORS
from keras import Model
from dotenv import load_dotenv

from music_classifier.inference import load_genre_model

from .routes import bp

load_dotenv()


def create_app(
        *,
        model: Model | None = None,
) -> Flask:
    """Create and configure the Flask application.

    Initializes the Flask backend, configures CORS support for frontend
    communication, loads the model, and registers API routes.

    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(__name__)

    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

    frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

    CORS(
        app, 
        resources={
            r"/*": {
                "origins": frontend_origin
            }
        },
    )

    if model is None:
        model = load_genre_model()

    app.config["MODEL"] = model

    app.register_blueprint(bp)

    return app
