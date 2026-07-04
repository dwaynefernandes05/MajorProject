#!/usr/bin/env python3
"""
Hugging Face Spaces Entry Point
================================

This script launches the Cable Specification Matcher on HF Spaces.
Automatically handles environment setup and model loading.
"""

import os
import sys
from pathlib import Path

# Set up environment for HF Spaces
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Import and launch the app
from app import create_interface

if __name__ == "__main__":
    print("🚀 Starting Cable Specification Matcher on Hugging Face Spaces...")
    
    try:
        demo = create_interface()
        
        # Launch with HF Spaces settings
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            analytics_enabled=False
        )
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        sys.exit(1)
