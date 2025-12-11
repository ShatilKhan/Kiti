#!/usr/bin/env python3
"""
Kiti Autonomous Vehicle - CLI Entry Point

This script provides a convenient command-line interface for running
the optical flow video processing pipeline for autonomous vehicles.
"""
import sys
from pathlib import Path

# Add src and project root directories to path
project_root = Path(__file__).parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import and run main
from main import main

if __name__ == '__main__':
    main()
