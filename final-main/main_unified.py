#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Entry Point for Python Interview System

This script serves as a single entry point for the entire Python interview system:
1. Checks and generates training data if not present
2. Checks and generates feedback dataset if not present
3. Trains the BERT classification model if not trained
4. Trains the BERT sequence-to-sequence model for feedback if not trained
5. Trains the T5 model for advanced feedback generation if requested
6. Runs the terminal interview system with the appropriate settings

Usage:
    python main_unified.py [--use-seq2seq] [--use-t5] [--use-combined]
"""

import os
import sys
import json
import argparse
import torch
from colorama import init, Fore, Style

# Import system components
import data_generator
import model_trainer
import model_trainer_seq2seq
import generate_feedback_data
from interview_system import InterviewSystem
from utils import ensure_directories_exist
from feedback_generator_t5 import FeedbackGeneratorT5
from feedback_orchestrator import FeedbackOrchestrator
# Import T5 model trainer (will be used conditionally)
import model_trainer_t5

# Initialize colorama for colored terminal output
init(autoreset=True)

def check_data_exists():
    """Check if training data exists."""
    return os.path.exists('data/training_data.json') and os.path.getsize('data/training_data.json') > 0

def check_feedback_data_exists():
    """Check if feedback dataset exists."""
    return os.path.exists('data/feedback_dataset.json') and os.path.getsize('data/feedback_dataset.json') > 0

def check_model_exists():
    """Check if trained model exists."""
    return os.path.exists('models/bert_interview_final.pt')

def check_feedback_model_exists():
    """Check if trained feedback model exists."""
    return os.path.exists('models/bert_feedback_final.pt')

def check_t5_model_exists():
    """Check if trained T5 model exists."""
    return os.path.exists('models/t5_feedback_final.pt')

def generate_data():
    """Generate training data if it doesn't exist."""
    print(f"{Fore.YELLOW}Checking training data...")
    
    if check_data_exists():
        print(f"{Fore.GREEN}Training data already exists.")
        return True
    
    print(f"{Fore.YELLOW}Training data not found. Generating training data...")
    try:
        # Call the data generation function from data_generator.py
        if hasattr(data_generator, 'main') and callable(data_generator.main):
            data_generator.main()
        else:
            print(f"{Fore.RED}Error: data_generator.py does not have a main() function.")
            return False
        
        if check_data_exists():
            print(f"{Fore.GREEN}Training data generated successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to generate training data.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error generating training data: {str(e)}")
        return False

def generate_feedback_dataset():
    """Generate feedback dataset if it doesn't exist or expand it."""
    print(f"{Fore.YELLOW}Checking feedback dataset...")
    
    if check_feedback_data_exists():
        print(f"{Fore.GREEN}Feedback dataset exists. Expanding it with more examples...")
    else:
        print(f"{Fore.YELLOW}Feedback dataset not found. Generating feedback dataset...")
    
    try:
        # Call the feedback data generation function
        generate_feedback_data.main()
        
        if check_feedback_data_exists():
            print(f"{Fore.GREEN}Feedback dataset generated/expanded successfully.")
            # Copy expanded dataset to feedback_dataset.json if it exists
            expanded_path = 'data/expanded_feedback_dataset.json'
            if os.path.exists(expanded_path):
                with open(expanded_path, 'r', encoding='utf-8') as f:
                    expanded_data = json.load(f)
                
                with open('data/feedback_dataset.json', 'w', encoding='utf-8') as f:
                    json.dump(expanded_data, f, indent=2)
                print(f"{Fore.GREEN}Expanded dataset copied to feedback_dataset.json")
            
            return True
        else:
            print(f"{Fore.RED}Error: Failed to generate feedback dataset.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error generating feedback dataset: {str(e)}")
        return False

def train_model():
    """Train the BERT model if it doesn't exist."""
    print(f"{Fore.YELLOW}Checking for trained model...")
    
    if check_model_exists():
        print(f"{Fore.GREEN}Trained model already exists.")
        return True
    
    print(f"{Fore.YELLOW}Trained model not found. Starting model training...")
    print(f"{Fore.YELLOW}This may take some time depending on your hardware.")
    
    try:
        # Call the model training function from model_trainer.py
        model_trainer.main()
        
        if check_model_exists():
            print(f"{Fore.GREEN}Model trained successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to train model.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error training model: {str(e)}")
        return False

def train_feedback_model():
    """Train the BERT sequence-to-sequence model for feedback generation."""
    print(f"{Fore.YELLOW}Checking for trained feedback model...")
    
    if check_feedback_model_exists():
        print(f"{Fore.GREEN}Trained feedback model already exists.")
        return True
    
    print(f"{Fore.YELLOW}Starting feedback model training...")
    print(f"{Fore.YELLOW}This may take some time depending on your hardware.")
    
    try:
        # Call the model training function from model_trainer_seq2seq.py
        model_trainer_seq2seq.main()
        
        if check_feedback_model_exists():
            print(f"{Fore.GREEN}Feedback model trained successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to train feedback model.")
            print(f"{Fore.YELLOW}The system will use rule-based feedback generation instead.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error training feedback model: {str(e)}")
        print(f"{Fore.YELLOW}The system will use rule-based feedback generation instead.")
        return False

def train_t5_model():
    """Train the T5 model for advanced feedback generation."""
    print(f"{Fore.YELLOW}Checking for trained T5 model...")
    
    if check_t5_model_exists():
        print(f"{Fore.GREEN}Trained T5 model already exists.")
        return True
    
    print(f"{Fore.YELLOW}Starting T5 model training...")
    print(f"{Fore.YELLOW}This may take some time depending on your hardware.")
    print(f"{Fore.YELLOW}Each epoch will be limited to 10 minutes of training time.")
    
    try:
        # Check if feedback dataset exists for training
        if not check_feedback_data_exists():
            print(f"{Fore.RED}Error: Feedback dataset not found. Cannot train T5 model.")
            print(f"{Fore.YELLOW}The system will use alternative feedback generation instead.")
            return False
            
        # Import and run the T5 model trainer
        import model_trainer_t5
        model_trainer_t5.main()
        
        if check_t5_model_exists():
            print(f"{Fore.GREEN}T5 model trained successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to train T5 model.")
            print(f"{Fore.YELLOW}The system will use alternative feedback generation instead.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error training T5 model: {str(e)}")
        print(f"{Fore.YELLOW}The system will use alternative feedback generation instead.")
        return False

def run_interview_system(use_seq2seq=False, use_t5=False, use_combined=False):
    """Run the terminal interview system."""
    print(f"{Fore.GREEN}Starting the Python Interview System...")
    
    try:
        # Verify that required data and models exist before starting
        if not check_data_exists():
            print(f"{Fore.RED}Error: Training data not found. Cannot run interview system.")
            return False
            
        if use_seq2seq and not check_feedback_model_exists():
            print(f"{Fore.YELLOW}Warning: Seq2Seq model not found but was requested.")
            print(f"{Fore.YELLOW}The system will fall back to rule-based feedback generation.")
            
        if use_t5 and not check_t5_model_exists():
            print(f"{Fore.YELLOW}Warning: T5 model not found but was requested.")
            print(f"{Fore.YELLOW}The system will fall back to alternative feedback generation.")
        
        # Initialize and run the interview system
        interview = InterviewSystem(use_seq2seq=use_seq2seq, use_t5=use_t5, use_combined=use_combined)
        interview.run()
        print(f"{Fore.GREEN}Interview system completed successfully.")
        return True
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\nInterview system interrupted by user.")
        return True
    except Exception as e:
        print(f"{Fore.RED}Error running interview system: {str(e)}")
        import traceback
        print(f"{Fore.RED}Traceback: {traceback.format_exc()}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Python Interview System with BERT-based Feedback')
    parser.add_argument('--use-seq2seq', action='store_true', help='Use the sequence-to-sequence model for feedback')
    parser.add_argument('--use-t5', action='store_true', help='Use the T5 model for advanced feedback generation')
    parser.add_argument('--use-combined', action='store_true', help='Use the combined orchestrator for comprehensive feedback')
    return parser.parse_args()

def main():
    """Main function to orchestrate the entire workflow."""
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}" + " " * 15 + "PYTHON INTERVIEW SYSTEM INITIALIZATION" + " " * 15)
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80 + "\n")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Ensure all required directories exist
        ensure_directories_exist()
        
        # Display system configuration
        print(f"{Fore.CYAN}System Configuration:")
        print(f"  - Using Seq2Seq model: {args.use_seq2seq}")
        print(f"  - Using T5 model: {args.use_t5}")
        print(f"  - Using Combined Orchestrator: {args.use_combined}")
        print(f"  - Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        
        # Step 1: Generate training data if needed
        if not generate_data():
            print(f"{Fore.RED}Failed to prepare training data. Exiting.")
            return False
        
        # Step 2: Generate or expand feedback dataset
        if not generate_feedback_dataset():
            print(f"{Fore.YELLOW}Failed to generate feedback dataset. Will use rule-based feedback generation.")
            if args.use_seq2seq or args.use_t5 or args.use_combined:
                print(f"{Fore.YELLOW}Warning: Advanced feedback models were requested but may not work properly without feedback dataset.")
        
        # Step 3: Train the classification model if needed
        if not train_model():
            print(f"{Fore.RED}Failed to train the model. Exiting.")
            return False
        
        # Step 4: Train the feedback model if needed and seq2seq is requested
        if args.use_seq2seq and not train_feedback_model():
            print(f"{Fore.YELLOW}Failed to train the feedback model. Will use rule-based feedback generation.")
        
        # Step 5: Train the T5 model if needed and T5 is requested
        if args.use_t5 and not train_t5_model():
            print(f"{Fore.YELLOW}Failed to train the T5 model. Will use alternative feedback generation.")
        
        # Verify all models are available as expected
        print(f"\n{Fore.CYAN}Model Status:")
        print(f"  - BERT Classification Model: {'Available' if check_model_exists() else 'Not Available'}")
        print(f"  - Seq2Seq Feedback Model: {'Available' if check_feedback_model_exists() else 'Not Available'}")
        print(f"  - T5 Feedback Model: {'Available' if check_t5_model_exists() else 'Not Available'}\n")
        
        # Step 6: Run the interview system
        if not run_interview_system(use_seq2seq=args.use_seq2seq, use_t5=args.use_t5, use_combined=args.use_combined):
            print(f"{Fore.RED}Failed to run the interview system. Exiting.")
            return False
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Python Interview System completed successfully.")
        return True
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}System interrupted by user.")
        return True
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error in main function: {str(e)}")
        import traceback
        print(f"{Fore.RED}Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    main()