import sys
import os

# Append current directory to sys.path so that 'import routing_core' works inside app.py
sys.path.append(os.path.dirname(__file__))

from .app import app
