#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Dataset Generator for Python Interview System

This script generates a dataset of Python interview questions, answers, and detailed
feedback for training the BERT sequence-to-sequence model. It expands the initial
feedback dataset with more examples.
"""

import os
import json
import random
from tqdm import tqdm

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Templates for generating detailed feedback
FEEDBACK_TEMPLATES = {
    'high_quality': [
        "Your answer correctly identifies {concept1}. You've also mentioned {concept2} which is important. To enhance your answer, you could also discuss {missing_concept} which would provide more depth to your explanation.",
        "Your response demonstrates a strong understanding of {concept1}. You've effectively explained {concept2}. For an even more comprehensive answer, consider including information about {missing_concept}.",
        "You've provided a thorough explanation of {concept1} and {concept2}. Your answer shows good knowledge of the topic. To make it even better, you might want to mention {missing_concept} as well."
    ],
    'medium_quality': [
        "Your answer covers {concept1}, which is good. However, you've missed some important aspects like {missing_concept1} and {missing_concept2}. Try to include these in your explanation to demonstrate a more complete understanding.",
        "You've mentioned {concept1} correctly, but your explanation of {concept2} could be more detailed. Also, consider discussing {missing_concept1} which is a key part of this topic.",
        "Your response shows a basic understanding of {concept1}. To improve, you should elaborate more on {concept2} and include information about {missing_concept1} which you didn't mention."
    ],
    'low_quality': [
        "Your answer touches on {concept1} but misses several critical aspects of the topic. You should focus on understanding {missing_concept1}, {missing_concept2}, and {missing_concept3} to improve your knowledge.",
        "While you mentioned {concept1}, your explanation contains some misconceptions. It's important to understand that {missing_concept1} is a key part of this topic, and you should also learn about {missing_concept2}.",
        "Your response shows only a basic familiarity with the topic. To develop a better understanding, focus on learning about {missing_concept1}, {missing_concept2}, and how they relate to {concept1}."
    ]
}

# Key concepts for different Python topics
KEY_CONCEPTS = {
    "list vs tuple": ["mutability", "syntax differences", "memory efficiency", "performance characteristics", "use cases", "dictionary keys", "immutability", "square brackets", "parentheses"],
    "memory management": ["reference counting", "garbage collection", "private heap", "circular references", "memory allocation", "memory deallocation", "memory pooling", "small objects"],
    "decorators": ["metaprogramming", "wrapper functions", "higher-order functions", "@syntax", "function modification", "nested functions", "closures", "function attributes"],
    "multiprocessing vs multithreading": ["GIL", "parallel execution", "memory space", "overhead", "inter-process communication", "I/O-bound tasks", "CPU-bound tasks", "shared memory"],
    "asyncio": ["coroutines", "event loop", "async/await syntax", "cooperative multitasking", "single-threaded concurrency", "I/O multiplexing", "task scheduling", "future objects"]
}

def load_initial_feedback_data(filepath='data/feedback_dataset.json'):
    """Load the initial feedback dataset."""
    if not os.path.exists(filepath):
        print(f"Warning: Initial feedback dataset {filepath} not found.")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_training_data(filepath='data/training_data.json'):
    """Load training data to extract more questions and answers."""
    if not os.path.exists(filepath):
        print(f"Warning: Training data file {filepath} not found.")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_feedback(question, answer, quality='high_quality'):
    """Generate detailed feedback for a question-answer pair."""
    # Determine the topic of the question
    topic = ""
    for key in KEY_CONCEPTS.keys():
        if key.lower() in question.lower():
            topic = key
            break
    
    if not topic:
        # Default to a generic topic if no specific match
        topic = "python concepts"
        concepts = ["syntax", "functionality", "implementation", "best practices", "efficiency"]
    else:
        concepts = KEY_CONCEPTS[topic]
    
    # Select random concepts for the template
    random.shuffle(concepts)
    concept1 = concepts[0] if len(concepts) > 0 else "the basic concept"
    concept2 = concepts[1] if len(concepts) > 1 else "implementation details"
    missing_concept1 = concepts[2] if len(concepts) > 2 else "advanced techniques"
    missing_concept2 = concepts[3] if len(concepts) > 3 else "edge cases"
    missing_concept3 = concepts[4] if len(concepts) > 4 else "performance considerations"
    
    # Select a random template based on quality
    template = random.choice(FEEDBACK_TEMPLATES[quality])
    
    # Fill in the template
    feedback = template.format(
        concept1=concept1,
        concept2=concept2,
        missing_concept=missing_concept1,
        missing_concept1=missing_concept1,
        missing_concept2=missing_concept2,
        missing_concept3=missing_concept3
    )
    
    # Add a score interpretation based on quality
    if quality == 'high_quality':
        score_range = random.randint(80, 95)
        feedback += f"\n\nYour answer demonstrates a strong understanding of the topic. Score: {score_range}/100"
    elif quality == 'medium_quality':
        score_range = random.randint(60, 79)
        feedback += f"\n\nYour answer shows a good understanding but has room for improvement. Score: {score_range}/100"
    else:  # low_quality
        score_range = random.randint(40, 59)
        feedback += f"\n\nYour answer needs significant improvement to fully address the question. Score: {score_range}/100"
    
    return feedback

def expand_feedback_dataset(initial_data, training_data, num_samples=50):
    """Expand the feedback dataset with more examples."""
    expanded_data = list(initial_data)  # Start with the initial data
    
    # Extract unique questions and their correct answers from training data
    question_answers = {}
    for item in training_data:
        question = item['question']
        if question not in question_answers and 'correct_answer' in item:
            question_answers[question] = item['correct_answer']
    
    # Generate additional feedback examples
    questions = list(question_answers.keys())
    random.shuffle(questions)
    
    for i in tqdm(range(min(num_samples, len(questions))), desc="Generating feedback examples"):
        question = questions[i]
        answer = question_answers[question]
        
        # Determine quality randomly, with a bias toward high quality
        quality = random.choices(
            ['high_quality', 'medium_quality', 'low_quality'],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]
        
        # Generate feedback
        feedback = generate_feedback(question, answer, quality)
        
        # Add to expanded dataset
        expanded_data.append({
            'question': question,
            'correct_answer': answer,
            'detailed_feedback': feedback
        })
    
    return expanded_data

def save_feedback_data(data, filename='feedback_dataset.json'):
    """Save feedback data to a JSON file."""
    filepath = os.path.join('data', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Expanded feedback dataset saved to {filepath}")

def main():
    print("Generating expanded feedback dataset for BERT sequence-to-sequence training...")
    
    # Load initial feedback data
    initial_data = load_initial_feedback_data()
    if not initial_data:
        print("Error: Initial feedback dataset not found. Please create it first.")
        return False
    
    # Load training data
    training_data = load_training_data()
    if not training_data:
        print("Error: Training data not found. Please generate it first.")
        return False
    
    # Expand the feedback dataset
    expanded_data = expand_feedback_dataset(initial_data, training_data, num_samples=50)
    
    # Save the expanded dataset
    save_feedback_data(expanded_data)
    
    print("Feedback dataset generation complete!")
    return True

if __name__ == "__main__":
    main()