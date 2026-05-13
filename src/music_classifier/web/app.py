"""Flask application factory and backend configuration."""

from flask import Flask
from flask_cors import CORS
from keras import Model

from music_classifier.inference import load_genre_model

from .routes import bp


def create_app(
        *,
        model: Model | None = None,
) -> Flask:
    """Create and configure the Flask application.

    Initializes the Flask backend, configures CORS support for
    frontend communication, and registers API routes.

    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(__name__)

    CORS(
        app, 
        resources={
            r"/*": {
                # Default Vite dev server origin.
                # Will update if frontend configuration differs.
                "origins": "http://localhost:5173"
            }
        },
    )

    if model is None:
        model = load_genre_model()

    app.config["MODEL"] = model

    app.register_blueprint(bp)

    return app
