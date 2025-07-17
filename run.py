#!/usr/bin/env python3
"""
Main entry point for the MNQ trading system.
Run from project root directory.
"""
import sys
import os

from src.main import main

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    main()