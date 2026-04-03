#!/usr/bin/env python3
"""Entry point for standalone invocation: ./ai/run.py or symlink as 'ai'."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.__main__ import main

main()
