"""Entry point WSGI dành cho cPanel Passenger."""

import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from a2wsgi import ASGIMiddleware
from app.server import app


application = ASGIMiddleware(app)
