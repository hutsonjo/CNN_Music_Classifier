"""Web backend package for the music genre classifier.

This package provides the Flask-based web backend used to expose the
inference engine through HTTP endpoints.

Components
----------
app
    Flask application factory and backend configuration.
routes
    API route handlers for health checks and genre prediction.

The backend is intentionally thin and delegates all inference logic to
the ``music_classifier.inference`` package.
"""

from .app import create_app
from .routes import predict

__all__ = [
    "create_app",
    "predict"
]