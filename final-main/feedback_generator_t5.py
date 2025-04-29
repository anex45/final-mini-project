#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
T5/BART-based Feedback Generator for Python Interview System

This module generates dynamic, detailed feedback for candidate responses using
a fine-tuned T5 or BART sequence-to-sequence model. These models are more advanced
than BERT for text generation tasks and can provide more contextual and nuanced feedback.
"""

import os
import json
import torch
import random
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from model_trainer import load_model

class FeedbackGeneratorT5:
    """Generates dynamic feedback for candidate responses using a T5 or BART model."""
    
    def __init__(self, model_type='t5', model_path=None, tokenizer_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        
        # Set default paths based on model type
        if model_path is None:
            model_path = f'models/{self.model_type}_feedback_final.pt'
        if tokenizer_path is None:
            tokenizer_path = f'models/{self.model_type}_feedback_tokenizer'
        
        # Load tokenizer and model based on model type
        self.initialize_tokenizer_and_model(model_path, tokenizer_path)

        # Load classification model for additional analysis
        self.classification_model = load_model('models/bert_interview_best.pt', self.device)
        self.classification_model.eval()
        
        # Load training data for reference answers
        self.training_data = self.load_training_data()
        
        # Initialize concept explanations database
        self.concept_explanations = self.initialize_concept_explanations()
        
        # Initialize code examples database
        self.code_examples = self.initialize_code_examples()
        
        # Understanding levels for dynamic feedback
        self.understanding_levels = {
            'high': ["excellent", "strong", "comprehensive", "thorough", "in-depth"],
            'medium': ["good", "solid", "adequate", "reasonable", "fair"],
            'low': ["basic", "limited", "partial", "developing", "emerging"]
        }
    
    def initialize_tokenizer_and_model(self, model_path, tokenizer_path):
        """Initialize the tokenizer and model based on model type."""
        try:
            # Initialize tokenizer
            if self.model_type == 't5':
                if os.path.exists(tokenizer_path):
                    self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
                else:
                    self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
                
                # Initialize model architecture
                self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
            elif self.model_type == 'bart':
                if os.path.exists(tokenizer_path):
                    self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
                else:
                    self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
                
                # Initialize model architecture
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}. Choose 't5' or 'bart'.")
            
            # Load trained weights if available
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded {self.model_type.upper()} model from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found. Using pretrained {self.model_type.upper()} model.")
                self.model.to(self.device)
                self.model.eval()
        except Exception as e:
            print(f"Error loading {self.model_type.upper()} model: {str(e)}")
            self.model = None
    
    def load_training_data(self, filepath='data/training_data.json'):
        """Load training data for reference answers."""
        if not os.path.exists(filepath):
            print(f"Warning: Training data file {filepath} not found.")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def initialize_concept_explanations(self):
        """Initialize database of concept explanations for feedback."""
        # This would be expanded with more concepts in a real implementation
        return {
            "inheritance": "Inheritance is a mechanism where a class can inherit attributes and methods from another class. The class that inherits is called a subclass or derived class, and the class being inherited from is called a superclass or base class.",
            "polymorphism": "Polymorphism allows objects of different classes to be treated as objects of a common superclass. It's often achieved through method overriding and enables more flexible and reusable code.",
            "encapsulation": "Encapsulation is the bundling of data and methods that operate on that data within a single unit (class). It restricts direct access to some of an object's components, which is a means of preventing unintended interference and misuse.",
            "decorator": "Decorators are a design pattern in Python that allow a user to add new functionality to an existing object without modifying its structure. They are usually called before the definition of a function you want to decorate.",
            "generator": "Generators are functions that can be paused and resumed, yielding multiple values over time instead of returning a single value and terminating. They use the 'yield' keyword and are memory-efficient for working with large datasets.",
            "list comprehension": "List comprehensions provide a concise way to create lists based on existing lists or other iterable objects. They consist of brackets containing an expression followed by a for clause, then zero or more for or if clauses.",
            "gil": "The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode at once. This simplifies memory management but can limit performance in CPU-bound and multi-threaded programs.",
            "asyncio": "Asyncio is a library to write concurrent code using the async/await syntax. It provides a framework for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources."
        }
    
    def initialize_code_examples(self):
        """Initialize database of code examples for feedback."""
        # This would be expanded with more examples in a real implementation
        return {
            "decorator": "# Example of a simple decorator\ndef my_decorator(func):\n    def wrapper():\n        print(\"Something is happening before the function is called.\")\n        func()\n        print(\"Something is happening after the function is called.\")\n    return wrapper\n\n@my_decorator\ndef say_hello():\n    print(\"Hello!\")\n\n# When we call say_hello(), the decorator is applied\nsay_hello()",
            
            "generator": "# Example of a generator function\ndef count_up_to(max):\n    count = 1\n    while count <= max:\n        yield count\n        count += 1\n\n# Using the generator\ncounter = count_up_to(5)\nfor number in counter:\n    print(number)  # Prints 1, 2, 3, 4, 5",
            
            "list comprehension": "# Example of list comprehension\nnumbers = [1, 2, 3, 4, 5]\n\n# Create a new list with squares of numbers\nsquares = [n**2 for n in numbers]\nprint(squares)  # Prints [1, 4, 9, 16, 25]\n\n# List comprehension with conditional\neven_squares = [n**2 for n in numbers if n % 2 == 0]\nprint(even_squares)  # Prints [4, 16]",
            
            "asyncio": "# Example of asyncio\nimport asyncio\n\nasync def fetch_data():\n    print(\"Start fetching\")\n    await asyncio.sleep(2)  # Simulating an I/O operation\n    print(\"Done fetching\")\n    return {\"data\": 42}\n\nasync def main():\n    task = asyncio.create_task(fetch_data())\n    print(\"Doing other work\")\n    await asyncio.sleep(1)\n    print(\"Still doing other work\")\n    result = await task\n    print(f\"Result: {result}\")\n\n# Run the async function\nasyncio.run(main())"
        }
    
    def find_reference_answers(self, question):
        """Find reference answers for a given question."""
        reference_answers = []
        
        for item in self.training_data:
            if item['question'] == question:
                reference_answers.append(item['correct_answer'])
        
        return reference_answers
    
    def get_model_embedding(self, text):
        """Get model embedding for a text."""
        # This is a simplified version - in a real implementation, you would use
        # the encoder part of the model to get embeddings
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if self.model_type == 't5':
                # For T5, we can use the encoder outputs
                outputs = self.model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                # Use the mean of the last hidden states as the embedding
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:  # BART
                # For BART, similar approach
                outputs = self.model.model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embedding
    
    def calculate_similarity(self, candidate_answer, reference_answer):
        """Calculate semantic similarity between candidate and reference answers."""
        candidate_embedding = self.get_model_embedding(candidate_answer)
        reference_embedding = self.get_model_embedding(reference_answer)
        
        similarity = cosine_similarity(candidate_embedding, reference_embedding)[0][0]
        return similarity
    
    def extract_key_concepts(self, answers):
        """Extract key concepts from reference answers."""
        # Enhanced implementation with more sophisticated NLP techniques
        all_text = " ".join(answers)
        words = all_text.lower().split()
        
        # Remove common words, punctuation, and apply more advanced filtering
        stopwords = {'and', 'the', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'that', 'this', 'are', 'as', 'be', 'by'}
        words = [word for word in words if len(word) > 3 and word.isalpha() and word not in stopwords]
        
        # Count word frequency with context awareness
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Extract phrases (bigrams) for more context-aware concepts
        phrases = []
        for i in range(len(words) - 1):
            if words[i] and words[i+1]:  # Ensure both words exist
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase)
        
        # Count phrase frequency
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Combine single words and meaningful phrases as key concepts
        word_concepts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:7]
        phrase_concepts = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Combine both types of concepts
        key_concepts = [concept[0] for concept in word_concepts] + [concept[0] for concept in phrase_concepts]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in key_concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
                
        return unique_concepts[:12]  # Return top concepts
    
    def identify_missing_concepts(self, candidate_answer, reference_answers):
        """Identify key concepts missing from the candidate's answer."""
        # Extract key concepts from reference answers
        reference_concepts = self.extract_key_concepts(reference_answers)
        
        # Check which concepts are missing from the candidate's answer
        missing_concepts = []
        for concept in reference_concepts:
            if concept.lower() not in candidate_answer.lower():
                missing_concepts.append(concept)
        
        return missing_concepts[:5]  # Return top 5 missing concepts
    
    def generate_feedback(self, question, candidate_answer):
        """Generate detailed feedback for a candidate's answer using the T5/BART model."""
        # If model is not available, use rule-based feedback
        if self.model is None:
            return self.generate_rule_based_feedback(question, candidate_answer)
        
        # Find reference answers
        reference_answers = self.find_reference_answers(question)
        if not reference_answers:
            return "I don't have reference answers for this question. Please check with your instructor."
        
        # Prepare input for the model
        if self.model_type == 't5':
            input_text = f"generate feedback: Question: {question} Answer: {candidate_answer}"
        else:  # BART
            input_text = f"Question: {question} Answer: {candidate_answer}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(self.device)
        
        # Generate feedback
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0
                )
            
            # Decode the generated feedback
            model_feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate similarity score for additional context
            similarity_scores = [self.calculate_similarity(candidate_answer, ref) for ref in reference_answers]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            
            # Enhance the model's feedback with additional information
            enhanced_feedback = self.enhance_feedback(model_feedback, question, candidate_answer, reference_answers, avg_similarity)
            
            return enhanced_feedback
        
        except Exception as e:
            print(f"Error generating feedback with {self.model_type.upper()} model: {str(e)}")
            # Fallback to rule-based feedback
            return self.generate_rule_based_feedback(question, candidate_answer)
    
    def enhance_feedback(self, model_feedback, question, candidate_answer, reference_answers, similarity_score):
        """Enhance the model-generated feedback with additional information."""
        # Identify missing concepts
        missing_concepts = self.identify_missing_concepts(candidate_answer, reference_answers)
        
        # Add score interpretation
        score_interpretation = self.get_score_interpretation(similarity_score)
        
        # Add concept explanations if relevant
        concept_explanations = ""
        for concept in missing_concepts[:2]:  # Add explanations for top 2 missing concepts
            if concept in self.concept_explanations:
                concept_explanations += f"\n\n{concept.capitalize()}: {self.concept_explanations[concept]}"
        
        # Add code example if relevant
        code_example = ""
        for concept in missing_concepts[:1]:  # Add code example for top missing concept
            if concept in self.code_examples:
                code_example = f"\n\nHere's an example of {concept}:\n{self.code_examples[concept]}"
        
        # Combine all parts
        enhanced_feedback = f"{model_feedback}\n\n{score_interpretation}{concept_explanations}{code_example}"
        
        return enhanced_feedback
    
    def get_score_interpretation(self, similarity_score):
        """Get interpretation of similarity score."""
        if similarity_score >= 0.8:
            level = "high"
            score_range = random.randint(85, 100)
            interpretation = random.choice([
                f"Your answer demonstrates an {random.choice(self.understanding_levels[level])} understanding of the topic.",
                f"You have an {random.choice(self.understanding_levels[level])} grasp of the key concepts.",
                f"Your response shows {random.choice(self.understanding_levels[level])} knowledge of the subject matter."
            ])
        elif similarity_score >= 0.6:
            level = "medium"
            score_range = random.randint(70, 84)
            interpretation = random.choice([
                f"Your answer shows a {random.choice(self.understanding_levels[level])} understanding of the topic.",
                f"You have a {random.choice(self.understanding_levels[level])} grasp of the key concepts.",
                f"Your response demonstrates {random.choice(self.understanding_levels[level])} knowledge of the subject matter."
            ])
        else:
            level = "low"
            score_range = random.randint(50, 69)
            interpretation = random.choice([
                f"Your answer shows a {random.choice(self.understanding_levels[level])} understanding of the topic.",
                f"You have a {random.choice(self.understanding_levels[level])} grasp of the key concepts.",
                f"Your response demonstrates {random.choice(self.understanding_levels[level])} knowledge of the subject matter."
            ])
        
        return f"{interpretation} Score: {score_range}/100"
    
    def generate_rule_based_feedback(self, question, candidate_answer):
        """Generate rule-based feedback when the model is not available."""
        # Find reference answers
        reference_answers = self.find_reference_answers(question)
        if not reference_answers:
            return "I don't have reference answers for this question. Please check with your instructor."
        
        # Calculate similarity scores
        similarity_scores = [self.calculate_similarity(candidate_answer, ref) for ref in reference_answers]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        # Identify missing concepts
        missing_concepts = self.identify_missing_concepts(candidate_answer, reference_answers)
        
        # Generate feedback based on similarity score
        if avg_similarity >= 0.8:
            quality = "high_quality"
        elif avg_similarity >= 0.6:
            quality = "medium_quality"
        else:
            quality = "low_quality"
        
        # Select feedback template based on quality
        if quality == "high_quality":
            feedback = random.choice([
                f"Your answer correctly identifies key concepts and shows a strong understanding of the topic. ",
                f"Your response demonstrates excellent knowledge of the subject matter. ",
                f"You've provided a comprehensive explanation that covers most of the important aspects. "
            ])
            
            if missing_concepts:
                feedback += f"To enhance your answer further, consider discussing {', '.join(missing_concepts[:2])}."
            else:
                feedback += "Your answer is very complete and covers all the key concepts."
                
        elif quality == "medium_quality":
            feedback = random.choice([
                f"Your answer shows a good understanding of the topic but could be more comprehensive. ",
                f"You've covered some important aspects, but your explanation could be more detailed. ",
                f"Your response demonstrates a solid grasp of the basics, but there's room for improvement. "
            ])
            
            if missing_concepts:
                feedback += f"Consider including information about {', '.join(missing_concepts[:3])} to make your answer more complete."
            else:
                feedback += "Try to provide more detailed explanations of the concepts you've mentioned."
                
        else:  # low_quality
            feedback = random.choice([
                f"Your answer shows a basic understanding of the topic but misses several important concepts. ",
                f"Your explanation is incomplete and needs significant improvement. ",
                f"Your response indicates some familiarity with the subject, but lacks depth and accuracy. "
            ])
            
            if missing_concepts:
                feedback += f"To improve, focus on understanding {', '.join(missing_concepts[:4])} which are key to this topic."
            else:
                feedback += "Try to deepen your understanding of the core concepts and provide more specific details."
        
        # Add score interpretation
        score_interpretation = self.get_score_interpretation(avg_similarity)
        feedback += f"\n\n{score_interpretation}"
        
        # Add concept explanations if relevant
        for concept in missing_concepts[:2]:  # Add explanations for top 2 missing concepts
            if concept in self.concept_explanations:
                feedback += f"\n\n{concept.capitalize()}: {self.concept_explanations[concept]}"
        
        # Add code example if relevant
        for concept in missing_concepts[:1]:  # Add code example for top missing concept
            if concept in self.code_examples:
                feedback += f"\n\nHere's an example of {concept}:\n{self.code_examples[concept]}"
        
        return feedback