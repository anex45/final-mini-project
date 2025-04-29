#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sequence-to-Sequence Model Trainer for Python Interview System

This module fine-tunes a BERT-based sequence-to-sequence model on the training dataset
of Python interview questions, answers, and detailed feedback. The model is trained to
generate detailed feedback for candidate responses.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    BertConfig,
    BertModel,
    BertLMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

class FeedbackDataset(Dataset):
    """Dataset for fine-tuning BERT on interview Q&A pairs with feedback."""
    
    def __init__(self, questions, answers, feedback, tokenizer, max_length=512, max_target_length=512):
        self.questions = questions
        self.answers = answers
        self.feedback = feedback
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        feedback = self.feedback[idx]
        
        # Combine question and answer as input
        input_text = f"Question: {question} Answer: {answer}"
        
        # Tokenize the input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            feedback,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels for the decoder (shifting right)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss calculation
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def load_feedback_data(filepath=None):
    """Load training data with feedback from JSON file."""
    if filepath is None:
        # Use absolute path based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, 'data', 'feedback_dataset.json')
    
    print(f"Loading feedback data from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    answers = []
    feedback = []
    
    for item in data:
        questions.append(item['question'])
        answers.append(item['correct_answer'])
        feedback.append(item['detailed_feedback'])
    
    return questions, answers, feedback

def initialize_model(tokenizer):
    """Initialize a BERT encoder-decoder model for sequence-to-sequence tasks."""
    # Initialize encoder and decoder with BERT configuration
    encoder_config = BertConfig.from_pretrained('bert-base-uncased')
    decoder_config = BertConfig.from_pretrained('bert-base-uncased')
    
    # Set decoder to use cross-attention and causal language modeling
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    
    # Initialize the encoder-decoder model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased', 
        encoder_config=encoder_config,
        decoder_config=decoder_config
    )
    
    # Set special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set beam search parameters
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    return model

def train_seq2seq_model(model, train_dataloader, val_dataloader, device, epochs=3):
    """Train the sequence-to-sequence model on the feedback dataset.
    Each epoch is strictly limited to 10 minutes of training time.
    """
    import time
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    total_training_time = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        processed_batches = 0
        
        # Set time limit for each epoch (10 minutes = 600 seconds)
        epoch_time_limit = 600  # seconds
        epoch_start_time = time.time()
        time_limit_reached = False
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Check if we've exceeded the time limit for this epoch
            elapsed_time = time.time() - epoch_start_time
            if elapsed_time >= epoch_time_limit:
                time_limit_reached = True
                print(f"\nReached 10-minute time limit for epoch {epoch+1}. Stopping training for this epoch.")
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_loss += loss.item()
            processed_batches += 1
            
            # Periodically check time to avoid going significantly over the limit
            if batch_idx % 5 == 0 and batch_idx > 0:
                current_time = time.time()
                if current_time - epoch_start_time >= epoch_time_limit:
                    time_limit_reached = True
                    print(f"\nReached 10-minute time limit for epoch {epoch+1}. Stopping training for this epoch.")
                    print(f"Time elapsed: {current_time - epoch_start_time:.2f} seconds")
                    break
        
        # Calculate actual epoch training time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration
        
        # Calculate training metrics
        if processed_batches > 0:
            train_loss = train_loss / processed_batches
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Processed {processed_batches}/{len(train_dataloader)} batches in this epoch")
            print(f"Epoch training time: {epoch_duration:.2f} seconds")
        else:
            train_loss = float('inf')  # Handle case where no batches were processed
            print("No batches processed in this epoch")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Track metrics
                val_loss += loss.item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_dataloader)
        val_duration = time.time() - val_start_time
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Validation time: {val_duration:.2f} seconds")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model with loss: {val_loss:.4f}")
            save_model(model, 'models/bert_feedback_best.pt')
        
        # If time limit was reached, inform about partial epoch completion
        if time_limit_reached:
            completion_percentage = (processed_batches / len(train_dataloader)) * 100
            print(f"Completed approximately {completion_percentage:.1f}% of epoch {epoch+1} before reaching time limit")
    
    print(f"\nTotal training time across all epochs: {total_training_time:.2f} seconds")
    return model

def save_model(model, filepath):
    """Save the trained model."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, tokenizer, device):
    """Load a trained model."""
    model = initialize_model(tokenizer)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_feedback(model, tokenizer, question, answer, device, max_length=512):
    """Generate feedback for a given question-answer pair."""
    # Combine question and answer as input
    input_text = f"Question: {question} Answer: {answer}"
    
    # Tokenize the input
    inputs = tokenizer(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate feedback
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0
    )
    
    # Decode the generated feedback
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return feedback

def main():
    print("Starting BERT sequence-to-sequence model fine-tuning for feedback generation...")
    print("Each epoch will be limited to 10 minutes of training time.")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize model
    model = initialize_model(tokenizer)
    model.to(device)
    
    # Load and prepare data
    questions, answers, feedback = load_feedback_data()
    
    # Split data into train and validation sets
    train_questions, val_questions, train_answers, val_answers, train_feedback, val_feedback = train_test_split(
        questions, answers, feedback, test_size=0.1, random_state=42
    )
    
    print(f"Training samples: {len(train_questions)}")
    print(f"Validation samples: {len(val_questions)}")
    
    # Create datasets
    train_dataset = FeedbackDataset(
        train_questions, train_answers, train_feedback, tokenizer
    )
    val_dataset = FeedbackDataset(
        val_questions, val_answers, val_feedback, tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,  # Smaller batch size for seq2seq training
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4
    )
    
    # Train model
    # Using default epochs=3 as each epoch is limited to 10 minutes
    # This provides a balance between training time and model performance
    model = train_seq2seq_model(model, train_dataloader, val_dataloader, device)
    
    # Save final model
    save_model(model, 'models/bert_feedback_final.pt')
    
    # Save tokenizer for later use
    tokenizer.save_pretrained('models/feedback_tokenizer')
    
    print("Model training complete!")

if __name__ == "__main__":
    main()