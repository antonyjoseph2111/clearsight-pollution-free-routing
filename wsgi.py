import sys
import os

# Add the 'app' directory to sys.path so that modules inside it (like routing_core) can be imported
# relative to that directory, consistent with how the app was developed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import the Flask app instance
from app.app import app

if __name__ == "__main__":
    app.run()
