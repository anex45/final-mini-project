#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Generator for Python Interview System

This module generates a training dataset of Python interview questions and answers
for fine-tuning the BERT model. The dataset includes various categories of questions
and multiple acceptable answers for each question.
"""

import os
import json
import random
from tqdm import tqdm
import pandas as pd

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Define question categories and templates
QUESTION_CATEGORIES = {
    'python_basics': [
        "What is the difference between a list and a tuple in Python?",
        "Explain how Python handles memory management.",
        "What are Python decorators and how do they work?",
        "Explain the concept of list comprehensions in Python.",
        "What is the Global Interpreter Lock (GIL) in Python?",
        "How does exception handling work in Python?",
        "What are Python generators and how do they differ from regular functions?",
        "Explain the difference between '==' and 'is' operators in Python.",
        "What are Python context managers and how do they work?",
        "Explain the concept of duck typing in Python."
    ],
    'data_structures': [
        "Explain the time complexity of common operations in Python dictionaries.",
        "What is the difference between a stack and a queue?",
        "How would you implement a linked list in Python?",
        "Explain the concept of hash tables and their implementation in Python.",
        "What are binary trees and how would you implement one in Python?",
        "Explain the difference between depth-first search and breadth-first search.",
        "What is a heap data structure and when would you use it?",
        "How would you implement a graph data structure in Python?",
        "Explain the concept of dynamic programming with an example.",
        "What are the advantages and disadvantages of using arrays vs linked lists?"
    ],
    'algorithms': [
        "Explain the quicksort algorithm and its time complexity.",
        "How would you find the kth largest element in an unsorted array?",
        "Explain the concept of recursion with an example.",
        "What is the difference between a greedy algorithm and dynamic programming?",
        "How would you detect a cycle in a linked list?",
        "Explain the concept of binary search and its time complexity.",
        "How would you implement a breadth-first search algorithm?",
        "What is the traveling salesman problem and how would you approach it?",
        "Explain the concept of backtracking with an example.",
        "How would you find the longest common subsequence of two strings?"
    ],
    'oop_concepts': [
        "Explain the concept of inheritance in Python with an example.",
        "What is polymorphism and how is it implemented in Python?",
        "Explain the concept of encapsulation in Python.",
        "What are abstract classes and interfaces in Python?",
        "Explain the difference between class methods, static methods, and instance methods.",
        "What is method overriding and how does it work in Python?",
        "Explain the concept of multiple inheritance and the method resolution order in Python.",
        "What are metaclasses in Python and how would you use them?",
        "Explain the concept of composition vs inheritance with examples.",
        "What is the __init__ method in Python and how is it used?"
    ],
    'advanced_python': [
        "Explain the asyncio module in Python and how it works.",
        "What are Python descriptors and how do they work?",
        "Explain the concept of metaprogramming in Python.",
        "What are Python's magic methods and how would you use them?",
        "Explain the concept of closures in Python with an example.",
        "What is the difference between multiprocessing and multithreading in Python?",
        "Explain the concept of coroutines in Python.",
        "What are Python type hints and how do they improve code quality?",
        "Explain the concept of monkey patching in Python and when it might be useful.",
        "What are Python's data model and how does it relate to special methods?"
    ]
}

# Generate answers for each question
def generate_answers():
    answers = {
        # Python Basics
        "What is the difference between a list and a tuple in Python?": [
            "Lists are mutable, meaning they can be modified after creation, while tuples are immutable. Lists use square brackets [] and tuples use parentheses (). Lists generally consume more memory than tuples and are slower for iteration.",
            "The main difference is mutability: lists can be changed after creation (mutable) while tuples cannot (immutable). Lists use square brackets and support operations like append() and extend(), while tuples use parentheses and don't allow modification after creation. Tuples are more memory efficient and slightly faster than lists.",
            "Lists are mutable collections that can be modified after creation, while tuples are immutable. Lists are defined using square brackets [], while tuples use parentheses (). Tuples are more memory-efficient and can be used as dictionary keys (since they're immutable), which lists cannot."
        ],
        "Explain how Python handles memory management.": [
            "Python uses automatic memory management with a private heap to store objects and data structures. It has a built-in garbage collector that reclaims memory from objects that are no longer referenced. Python uses reference counting as its primary memory management technique, along with cycle-detecting garbage collection to handle circular references.",
            "Python manages memory through automatic garbage collection. It uses reference counting to track how many references point to an object, and when the count reaches zero, the memory is freed. Python also has a cycle detector to handle circular references. All Python objects are stored in a private heap, and the memory manager allocates heap space for objects.",
            "Python uses automatic memory management through reference counting and garbage collection. When objects are created, Python allocates memory from a private heap. It tracks references to objects, and when an object's reference count drops to zero, the memory is reclaimed. Python also has a cyclic garbage collector to detect and collect objects in reference cycles that would otherwise not be collected."
        ],
        # More answers would be added for each question in the actual implementation
    }
    return answers

def generate_training_data(num_samples=1000):
    """Generate training data for fine-tuning the BERT model."""
    answers_dict = generate_answers()
    
    # For questions without predefined answers, generate placeholder answers
    all_questions = [q for category in QUESTION_CATEGORIES.values() for q in category]
    for question in all_questions:
        if question not in answers_dict:
            # Generate placeholder answers for questions without predefined answers
            answers_dict[question] = [
                f"Sample answer 1 for: {question}",
                f"Sample answer 2 for: {question}",
                f"Sample answer 3 for: {question}"
            ]
    
    # Generate training samples
    training_data = []
    for _ in tqdm(range(num_samples), desc="Generating training data"):
        # Randomly select a question
        category = random.choice(list(QUESTION_CATEGORIES.keys()))
        question = random.choice(QUESTION_CATEGORIES[category])
        
        # Get answers for the question
        answers = answers_dict[question]
        correct_answer = random.choice(answers)
        
        # Generate incorrect answers by selecting from other questions
        other_questions = [q for q in all_questions if q != question]
        incorrect_answers = []
        for _ in range(3):  # Generate 3 incorrect answers
            other_question = random.choice(other_questions)
            incorrect_answer = random.choice(answers_dict[other_question])
            incorrect_answers.append(incorrect_answer)
        
        # Add sample to training data
        training_data.append({
            'question': question,
            'category': category,
            'correct_answer': correct_answer,
            'incorrect_answers': incorrect_answers
        })
    
    return training_data

def save_training_data(data, filename='training_data.json'):
    """Save training data to a JSON file."""
    filepath = os.path.join('data', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Training data saved to {filepath}")
    
    # Also save as CSV for easier viewing
    df = pd.DataFrame([
        {
            'question': item['question'],
            'category': item['category'],
            'correct_answer': item['correct_answer'],
            'incorrect_answer_1': item['incorrect_answers'][0] if len(item['incorrect_answers']) > 0 else '',
            'incorrect_answer_2': item['incorrect_answers'][1] if len(item['incorrect_answers']) > 1 else '',
            'incorrect_answer_3': item['incorrect_answers'][2] if len(item['incorrect_answers']) > 2 else ''
        } for item in data
    ])
    
    csv_filepath = os.path.join('data', 'training_data.csv')
    df.to_csv(csv_filepath, index=False)
    print(f"Training data also saved as CSV to {csv_filepath}")

def generate_evaluation_data(num_samples=100):
    """Generate evaluation data for testing the model."""
    # Similar to training data but with fewer samples
    return generate_training_data(num_samples)

def main():
    print("Generating Python interview training dataset...")
    
    # Generate training data
    training_data = generate_training_data(num_samples=1000)
    save_training_data(training_data, 'training_data.json')
    
    # Generate evaluation data
    eval_data = generate_evaluation_data(num_samples=100)
    save_training_data(eval_data, 'evaluation_data.json')
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()