import os
import sys

# Add src to sys.path so tests can import from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
