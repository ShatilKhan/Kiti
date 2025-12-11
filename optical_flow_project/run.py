#!/usr/bin/env python3
"""
Optical Flow Video Processing - CLI Entry Point

This script provides a convenient command-line interface for running
the optical flow video processing pipeline.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Import and run main
from main import main

if __name__ == '__main__':
    main()
