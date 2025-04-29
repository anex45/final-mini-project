#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Python Interview System

This module provides helper functions used across the interview system.
"""

import os
import json
import torch
from transformers import BertTokenizer

def ensure_directories_exist():
    """Ensure that all required directories exist."""
    for directory in ['data', 'models', 'sessions']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_json_data(filepath):
    """Load data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_data(data, filepath):
    """Save data to a JSON file."""
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filepath}")

def get_device():
    """Get the appropriate device for PyTorch operations."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(tokenizer_path='models/tokenizer'):
    """Load the BERT tokenizer."""
    if os.path.exists(tokenizer_path):
        return BertTokenizer.from_pretrained(tokenizer_path)
    else:
        return BertTokenizer.from_pretrained('bert-base-uncased')

def truncate_text(text, max_length=50):
    """Truncate text to a maximum length and add ellipsis if needed."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def format_category_name(category):
    """Format a category name for display (replace underscores with spaces and capitalize)."""
    return category.replace('_', ' ').title()

def get_multiline_input(prompt="> "):
    """Get multiline input from the user."""
    print(prompt, end="")
    lines = []
    while True:
        line = input()
        if line.lower() == 'exit':
            return 'exit'
        if line.strip() == '' and lines and lines[-1].strip() == '':
            break
        lines.append(line)
    
    return '\n'.join(lines)