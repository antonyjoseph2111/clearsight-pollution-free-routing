# Expose the Flask app object so that 'from app import app' works.
# This allows 'gunicorn app:app' (Render's default guess) to succeed.
from .app import app
