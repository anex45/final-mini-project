#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT-based Feedback Generator for Python Interview System

This module generates dynamic, detailed feedback for candidate responses using
a fine-tuned BERT sequence-to-sequence model. The feedback is contextual and
provides specific suggestions for improvement.
"""

import os
import json
import torch
import random
import numpy as np
from transformers import BertTokenizer, EncoderDecoderModel
from sklearn.metrics.pairwise import cosine_similarity
from model_trainer import load_model

class FeedbackGeneratorSeq2Seq:
    """Generates dynamic feedback for candidate responses using a seq2seq model."""
    
    def __init__(self, model_path='models/bert_feedback_final.pt', tokenizer_path='models/feedback_tokenizer'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path if os.path.exists(tokenizer_path) else 'bert-base-uncased')
        
        # Initialize and load the sequence-to-sequence model
        self.initialize_model(model_path)

        # Load classification model
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
    
    def initialize_model(self, model_path):
        """Initialize the sequence-to-sequence model for feedback generation."""
        # Initialize encoder-decoder configuration
        self.model = None
        
        if os.path.exists(model_path):
            try:
                # Initialize the model architecture
                self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                    'bert-base-uncased', 'bert-base-uncased'
                )
                
                # Set special tokens
                self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
                self.model.config.eos_token_id = self.tokenizer.sep_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
                # Set beam search parameters
                self.model.config.max_length = 512
                self.model.config.early_stopping = True
                self.model.config.no_repeat_ngram_size = 3
                self.model.config.length_penalty = 2.0
                self.model.config.num_beams = 4
                
                # Load the trained weights
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded seq2seq model from {model_path}")
            except Exception as e:
                print(f"Error loading seq2seq model: {str(e)}")
                self.model = None
        else:
            print(f"Warning: Model file {model_path} not found. Using fallback feedback generation.")
    
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
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for a text using the encoder part of our model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use the encoder part of the seq2seq model to get embeddings
            outputs = self.model.encoder(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding
    
    def calculate_similarity(self, candidate_answer, reference_answer):
        """Calculate semantic similarity between candidate and reference answers."""
        candidate_embedding = self.get_bert_embedding(candidate_answer)
        reference_embedding = self.get_bert_embedding(reference_answer)
        
        similarity = cosine_similarity(candidate_embedding, reference_embedding)[0][0]
        return similarity
    
    def extract_key_concepts(self, answers):
        """Extract key concepts from reference answers using advanced NLP techniques."""
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
        key_concepts = self.extract_key_concepts(reference_answers)
        missing_concepts = []
        
        for concept in key_concepts:
            if concept.lower() not in candidate_answer.lower():
                missing_concepts.append(concept)
        
        return missing_concepts[:5]  # Return top 5 missing concepts
    
    def get_concept_explanation(self, concept, reference_answers=None):
        """Get an explanation for a concept from the database or generate one."""
        # Check if we have a pre-defined explanation
        for key, explanation in self.concept_explanations.items():
            if key in concept.lower():
                return explanation
        
        # If no specific explanation found, extract from reference answers
        if reference_answers:
            # Find sentences containing the concept
            all_text = " ".join(reference_answers)
            sentences = all_text.split('.')
            relevant_sentences = [s for s in sentences if concept.lower() in s.lower()]
            
            if relevant_sentences:
                return relevant_sentences[0].strip() + "."
        
        # If no explanation found, return a generic one
        return f"This concept involves understanding {concept} in the context of Python programming."
    
    def get_code_example(self, concept):
        """Get a code example for a given concept if available."""
        # Check if we have a pre-defined code example
        for key, example in self.code_examples.items():
            if key in concept.lower():
                return example
        
        # If no specific example found, return None
        return None
    
    def calculate_nuanced_score(self, similarity, missing_concept_count, mentioned_concept_count):
        """Calculate a more nuanced score on a scale of 0-100."""
        # Base score from similarity (0-70 points)
        similarity_score = min(70, similarity * 70)
        
        # Points for mentioned concepts (0-20 points)
        concept_score = min(20, mentioned_concept_count * 4)
        
        # Penalty for missing concepts (0-15 points)
        missing_penalty = min(15, missing_concept_count * 3)
        
        # Calculate final score
        final_score = similarity_score + concept_score - missing_penalty
        
        # Ensure score is within 0-100 range
        final_score = max(0, min(100, final_score))
        
        # Round to nearest integer
        return round(final_score)
    
    def get_score(self, question, candidate_answer):
        """Calculate a score for the candidate's answer on a scale of 0-100."""
        # Find reference answers for the question
        reference_answers = self.find_reference_answers(question)
        
        if not reference_answers:
            return 50  # Default score if no reference answers available
        
        # Calculate similarity scores
        similarities = [self.calculate_similarity(candidate_answer, ref) for ref in reference_answers]
        max_similarity = max(similarities)
        
        # Identify missing concepts
        missing_concepts = self.identify_missing_concepts(candidate_answer, reference_answers)
        
        # Identify mentioned concepts
        key_concepts = self.extract_key_concepts(reference_answers)
        mentioned_concepts = [concept for concept in key_concepts 
                            if concept.lower() in candidate_answer.lower()]
        
        # Calculate final score
        nuanced_score = self.calculate_nuanced_score(max_similarity, len(missing_concepts), len(mentioned_concepts))
        
        return nuanced_score
    
    def generate_feedback_with_seq2seq(self, question, candidate_answer):
        """Generate detailed feedback using the sequence-to-sequence model."""
        if self.model is None:
            return None
        
        try:
            # Combine question and answer as input
            input_text = f"Question: {question} Answer: {candidate_answer}"
            
            # Tokenize the input
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate feedback
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
            
            # Decode the generated feedback
            feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return feedback
        except Exception as e:
            print(f"Error generating feedback with seq2seq model: {str(e)}")
            return None
    
    def generate_feedback(self, question, candidate_answer):
        """Generate dynamic feedback for a candidate's answer."""
        # Try to generate feedback using the seq2seq model first
        seq2seq_feedback = self.generate_feedback_with_seq2seq(question, candidate_answer)
        
        if seq2seq_feedback:
            return seq2seq_feedback
        
        # Fall back to the rule-based approach if seq2seq fails
        print("Falling back to rule-based feedback generation")
        
        # Find reference answers for the question
        reference_answers = self.find_reference_answers(question)
        
        if not reference_answers:
            return "I don't have reference answers for this question yet."

        # Initialize feedback parts list
        feedback_parts = []
        
        # Classify answer using classification model
        input_text = f"{question} [SEP] {candidate_answer}"
        inputs = self.tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.classification_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0][1].item()
            prediction = logits.argmax(-1).item()

        # Add classification result to feedback
        feedback_parts.append(f"Classification: {'Correct' if prediction == 1 else 'Incorrect'} (Confidence: {confidence:.2%})")
        
        # Calculate similarity scores
        similarities = [self.calculate_similarity(candidate_answer, ref) for ref in reference_answers]
        max_similarity = max(similarities)
        
        # Determine understanding level
        if max_similarity > 0.85:
            level_category = 'high'
        elif max_similarity > 0.65:
            level_category = 'medium'
        else:
            level_category = 'low'
        
        understanding_level = random.choice(self.understanding_levels[level_category])
        
        # Identify missing concepts
        missing_concepts = self.identify_missing_concepts(candidate_answer, reference_answers)
        
        # Generate feedback components
        feedback_parts = []
        
        # Add strength feedback
        if max_similarity > 0.5:
            # Extract a concept that was mentioned correctly
            key_concepts = self.extract_key_concepts(reference_answers)
            mentioned_concepts = [concept for concept in key_concepts 
                                if concept.lower() in candidate_answer.lower()]
            
            if mentioned_concepts:
                concept = random.choice(mentioned_concepts)
                feedback_parts.append(f"Your response demonstrates a good understanding of {concept}.")
        
        # Add improvement feedback for missing concepts
        if missing_concepts:
            missing_concept = missing_concepts[0]
            feedback_parts.append(f"Your answer could be enhanced by including {missing_concept}.")
            
            # Add detailed explanation of the missing concept
            concept_explanation = self.get_concept_explanation(missing_concept, reference_answers)
            feedback_parts.append(f"{concept_explanation}")
            
            # Add code example if available
            code_example = self.get_code_example(missing_concept)
            if code_example:
                feedback_parts.append(f"Here's an example demonstrating {missing_concept}:\n```python\n{code_example}\n```")
        
        # Add conclusion
        topic = question.split('?')[0].lower()
        if len(topic) > 50:  # Truncate long topics
            topic = topic[:47] + '...'
        
        conclusion = f"Overall, your response shows {understanding_level} understanding of {topic}."
        feedback_parts.append(conclusion)
        
        # Calculate score
        score = self.get_score(question, candidate_answer)
        feedback_parts.append(f"\nScore: {score}/100")
        
        # Join all feedback parts with appropriate spacing
        feedback = "\n\n".join(feedback_parts)
        return feedback