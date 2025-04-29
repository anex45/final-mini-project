#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Trainer for Python Interview System

This module fine-tunes a BERT model on the training dataset of Python interview
questions and answers. The model is trained to evaluate the quality of candidate
responses by comparing them to reference answers.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

class InterviewDataset(Dataset):
    """Dataset for fine-tuning BERT on interview Q&A pairs."""
    
    def __init__(self, questions, answers, labels, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        label = self.labels[idx]
        
        # Tokenize the question-answer pair
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_training_data(filepath='data/training_data.json'):
    """Load training data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    correct_answers = []
    incorrect_answers = []
    
    for item in data:
        question = item['question']
        correct_answer = item['correct_answer']
        
        # Add correct answer pair
        questions.append(question)
        correct_answers.append(correct_answer)
        
        # Add incorrect answer pairs
        for incorrect_answer in item['incorrect_answers']:
            questions.append(question)
            incorrect_answers.append(incorrect_answer)
    
    # Create labels: 1 for correct answers, 0 for incorrect answers
    answers = correct_answers + incorrect_answers
    labels = [1] * len(correct_answers) + [0] * len(incorrect_answers)
    
    return questions, answers, labels

def train_model(model, train_dataloader, val_dataloader, device, epochs=3):
    """Train the BERT model on the interview dataset."""
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_dataloader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='binary', pos_label=1)
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # Track metrics
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_dataloader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
        val_precision = precision_score(val_labels, val_preds, average='binary', pos_label=1)
        val_recall = recall_score(val_labels, val_preds, average='binary', pos_label=1)

        
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"New best model with F1: {val_f1:.4f}")
            save_model(model, 'models/bert_interview_best.pt')
    
    return model

def save_model(model, filepath):
    """Save the trained model."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device):
    """Load a trained model."""
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    print("Starting BERT model fine-tuning for Python interview system...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Binary classification: correct/incorrect
    )
    model.to(device)
    
    # Load and prepare data
    questions, answers, labels = load_training_data()
    
    # Split data into train and validation sets
    train_questions, val_questions, train_answers, val_answers, train_labels, val_labels = train_test_split(
        questions, answers, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_questions)}")
    print(f"Validation samples: {len(val_questions)}")
    
    # Create datasets
    train_dataset = InterviewDataset(
        train_questions, train_answers, train_labels, tokenizer
    )
    val_dataset = InterviewDataset(
        val_questions, val_answers, val_labels, tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8
    )
    
    # Train model
    model = train_model(model, train_dataloader, val_dataloader, device, epochs=3)
    
    # Save final model
    save_model(model, 'models/bert_interview_final.pt')
    
    # Save tokenizer for later use
    tokenizer.save_pretrained('models/tokenizer')
    
    print("Model training complete!")

if __name__ == "__main__":
    main()