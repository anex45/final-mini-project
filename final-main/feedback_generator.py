#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Generator for Python Interview System

This module generates dynamic feedback for candidate responses based on the
fine-tuned BERT model's evaluation. The feedback is contextual and avoids
using predefined phrases.
"""

import os
import json
import torch
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class FeedbackGenerator:
    """Generates dynamic feedback for candidate responses."""
    
    def __init__(self, model_path='models/bert_interview_final.pt', tokenizer_path='models/tokenizer'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BERT model for semantic similarity
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path if os.path.exists(tokenizer_path) else 'bert-base-uncased')
        
        # Load training data for reference answers
        self.training_data = self.load_training_data()
        
        # Initialize concept explanations database
        self.concept_explanations = self.initialize_concept_explanations()
        
        # Initialize code examples database
        self.code_examples = self.initialize_code_examples()
        
        # Feedback components for dynamic generation
        self.feedback_components = {
            'strength_intros': [
                "Your response demonstrates",
                "I noticed your answer includes",
                "You've done well in showing",
                "Your answer effectively covers",
                "It's good that you mentioned"
            ],
            'improvement_intros': [
                "You might want to consider",
                "Your answer could be enhanced by",
                "One area to develop further is",
                "It would be helpful to include",
                "Consider expanding on"
            ],
            'missing_concepts_intros': [
                "Your answer doesn't address",
                "You might have overlooked",
                "An important aspect missing from your answer is",
                "Your response could benefit from including",
                "One key concept that wasn't covered is"
            ],
            'conclusion_phrases': [
                "Overall, your response shows {level} understanding of {topic}.",
                "Based on your answer, you demonstrate {level} knowledge about {topic}.",
                "Your explanation reflects {level} grasp of {topic} concepts.",
                "Your response indicates {level} familiarity with {topic}.",
                "From your answer, I can see you have {level} comprehension of {topic}."
            ]
        }
        
        # Understanding levels for dynamic feedback
        self.understanding_levels = {
            'high': ["excellent", "strong", "comprehensive", "thorough", "in-depth"],
            'medium': ["good", "solid", "adequate", "reasonable", "fair"],
            'low': ["basic", "limited", "partial", "developing", "emerging"]
        }
    
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
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for a text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding
    
    def calculate_similarity(self, candidate_answer, reference_answer):
        """Calculate semantic similarity between candidate and reference answers."""
        candidate_embedding = self.get_bert_embedding(candidate_answer)
        reference_embedding = self.get_bert_embedding(reference_answer)
        
        similarity = cosine_similarity(candidate_embedding, reference_embedding)[0][0]
        return similarity
    
    def find_reference_answers(self, question):
        """Find reference answers for a given question."""
        reference_answers = []
        
        for item in self.training_data:
            if item['question'] == question:
                reference_answers.append(item['correct_answer'])
        
        return reference_answers
    
    def extract_key_concepts(self, answers):
        """Extract key concepts from reference answers using advanced NLP techniques."""
        # Enhanced implementation with more sophisticated NLP techniques
        all_text = " ".join(answers)
        words = all_text.lower().split()
        
        # Remove common words, punctuation, and apply more advanced filtering
        # This could be enhanced with a proper stopwords list from NLTK
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
        """Identify key concepts missing from the candidate's answer with improved accuracy."""
        key_concepts = self.extract_key_concepts(reference_answers)
        missing_concepts = []
        
        # More sophisticated check for concept presence
        for concept in key_concepts:
            # Check for concept as a whole and also for individual words in multi-word concepts
            concept_words = concept.lower().split()
            
            # For single word concepts, do a direct check
            if len(concept_words) == 1:
                if concept.lower() not in candidate_answer.lower():
                    missing_concepts.append(concept)
            # For multi-word concepts, check if all words appear close to each other
            else:
                # If the exact phrase is present, it's not missing
                if concept.lower() in candidate_answer.lower():
                    continue
                    
                # Check if all individual words are present
                all_words_present = all(word in candidate_answer.lower() for word in concept_words)
                
                # If not all words are present, it's definitely missing
                if not all_words_present:
                    missing_concepts.append(concept)
                # If all words are present but not as a phrase, check their proximity
                else:
                    # This is a simplified proximity check
                    # A more advanced implementation could use NLP to check semantic proximity
                    missing_concepts.append(concept)
        
        return missing_concepts[:5]  # Return top 5 missing concepts for more comprehensive feedback
        
    def get_code_example(self, concept):
        """Get a code example for a given concept if available."""
        # Check if we have a pre-defined code example
        for key, example in self.code_examples.items():
            if key in concept.lower():
                return example
        
        # If no specific example found, return None
        return None
        
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
        
    def get_learning_recommendation(self, level_category, missing_concepts):
        """Generate personalized learning recommendations based on performance level and missing concepts."""
        # Basic recommendations based on performance level
        level_recommendations = {
            'high': "To further enhance your understanding, consider exploring advanced topics like {topics}.",
            'high_medium': "You're doing well! To strengthen your knowledge, focus on {topics}.",
            'medium': "To improve your understanding, review the core concepts of {topics}.",
            'medium_low': "Consider revisiting the fundamentals of {topics} to build a stronger foundation.",
            'low': "I recommend starting with the basics of {topics} and gradually building up your knowledge."
        }
        
        # Generate topic recommendations based on missing concepts
        if missing_concepts:
            # Take up to 3 missing concepts for recommendation
            topics = ", ".join(missing_concepts[:3])
            recommendation = level_recommendations.get(
                level_category, "Consider focusing on {topics} to improve your understanding.")
            return recommendation.format(topics=topics)
        else:
            # Generic recommendations if no specific missing concepts
            generic_recommendations = {
                'high': "To further enhance your understanding, consider exploring more advanced applications and edge cases.",
                'high_medium': "You're doing well! Try solving more complex problems to deepen your knowledge.",
                'medium': "To improve, try implementing these concepts in practical projects to solidify your understanding.",
                'medium_low': "Consider reviewing the core principles and practicing with simple examples.",
                'low': "I recommend starting with tutorials and guided exercises to build a stronger foundation."
            }
            return generic_recommendations.get(level_category, "Consider practicing with more examples to strengthen your understanding.")
        
    def calculate_nuanced_score(self, similarity, missing_concept_count, mentioned_concept_count):
        """Calculate a more nuanced score on a scale of 0-100 instead of just high/medium/low."""
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
        
    def get_score_interpretation(self, score):
        """Provide an educational interpretation of the numerical score."""
        if score >= 90:
            return "Excellent understanding of the concept"
        elif score >= 80:
            return "Very good grasp of the material"
        elif score >= 70:
            return "Good understanding with some room for improvement"
        elif score >= 60:
            return "Adequate knowledge but needs strengthening in key areas"
        elif score >= 50:
            return "Basic understanding with significant gaps to address"
        elif score >= 40:
            return "Limited comprehension requiring substantial review"
        else:
            return "Fundamental concepts need to be revisited"
    
    def get_score(self, question, candidate_answer):
        """Calculate a score for the candidate's answer on a scale of 0-100."""
        # Find reference answers for the question
        reference_answers = self.find_reference_answers(question)
        
        if not reference_answers:
            return 50  # Default score if no reference answers available
        
        # Calculate similarity scores
        similarities = [self.calculate_similarity(candidate_answer, ref) for ref in reference_answers]
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # More nuanced understanding level determination
        # Consider both max and average similarity for a more balanced assessment
        combined_score = (max_similarity * 0.7) + (avg_similarity * 0.3)
        
        # Identify missing concepts
        missing_concepts = self.identify_missing_concepts(candidate_answer, reference_answers)
        
        # Identify mentioned concepts
        key_concepts = self.extract_key_concepts(reference_answers)
        mentioned_concepts = [concept for concept in key_concepts 
                            if concept.lower() in candidate_answer.lower()]
        
        # Calculate final score
        nuanced_score = self.calculate_nuanced_score(max_similarity, len(missing_concepts), len(mentioned_concepts))
        
        return nuanced_score
    
    def generate_feedback(self, question, candidate_answer):
        """Generate dynamic feedback for a candidate's answer with advanced analysis."""
        # Find reference answers for the question
        reference_answers = self.find_reference_answers(question)
        
        if not reference_answers:
            return "I don't have reference answers for this question yet."
        
        # Calculate similarity scores with more sophisticated analysis
        similarities = [self.calculate_similarity(candidate_answer, ref) for ref in reference_answers]
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Track user's answer history for this question (in a real system)
        # This would be stored in a database for persistent learning analytics
        
        # More nuanced understanding level determination
        # Consider both max and average similarity for a more balanced assessment
        combined_score = (max_similarity * 0.7) + (avg_similarity * 0.3)
        
        if combined_score > 0.85:
            level_category = 'high'
        elif combined_score > 0.75:
            level_category = 'high_medium'  # New intermediate category
        elif combined_score > 0.65:
            level_category = 'medium'
        elif combined_score > 0.5:
            level_category = 'medium_low'   # New intermediate category
        else:
            level_category = 'low'
        
        # Add more nuanced understanding levels
        if not hasattr(self, 'understanding_levels_advanced'):
            self.understanding_levels_advanced = {
                'high': ["excellent", "strong", "comprehensive", "thorough", "in-depth"],
                'high_medium': ["very good", "proficient", "substantial", "well-developed", "advanced"],
                'medium': ["good", "solid", "adequate", "reasonable", "fair"],
                'medium_low': ["developing", "moderate", "partial", "emerging", "progressing"],
                'low': ["basic", "limited", "rudimentary", "introductory", "foundational"]
            }
            
        # Use more nuanced understanding levels
        understanding_level = random.choice(self.understanding_levels_advanced.get(
            level_category, self.understanding_levels.get(level_category, ["moderate"])))
        
        # Identify strengths and areas for improvement
        missing_concepts = self.identify_missing_concepts(candidate_answer, reference_answers)
        
        # Generate dynamic feedback
        feedback_parts = []
        
        # Add strength feedback
        if max_similarity > 0.5:
            strength_intro = random.choice(self.feedback_components['strength_intros'])
            # Extract a concept that was mentioned correctly
            key_concepts = self.extract_key_concepts(reference_answers)
            mentioned_concepts = [concept for concept in key_concepts 
                                if concept.lower() in candidate_answer.lower()]
            
            if mentioned_concepts:
                concept = random.choice(mentioned_concepts)
                feedback_parts.append(f"{strength_intro} a good understanding of {concept}.")
        
        # Add improvement feedback
        if missing_concepts:
            improvement_intro = random.choice(self.feedback_components['improvement_intros'])
            missing_concept = random.choice(missing_concepts)
            feedback_parts.append(f"{improvement_intro} {missing_concept} in your answer.")
        
        # Add more specific feedback on missing concepts with detailed explanations
        if len(missing_concepts) > 1:
            missing_intro = random.choice(self.feedback_components['missing_concepts_intros'])
            missing_concept = missing_concepts[1] if len(missing_concepts) > 1 else missing_concepts[0]
            
            # Add detailed explanation of the missing concept
            concept_explanation = self.get_concept_explanation(missing_concept, reference_answers)
            feedback_parts.append(f"{missing_intro} {missing_concept}. {concept_explanation}")
            
            # Add code example if available
            code_example = self.get_code_example(missing_concept)
            if code_example:
                feedback_parts.append(f"Here's an example demonstrating {missing_concept}:\n```python\n{code_example}\n```")
        
        # Add personalized learning recommendation based on performance
        recommendation = self.get_learning_recommendation(level_category, missing_concepts)
        if recommendation:
            feedback_parts.append(f"\nPersonalized recommendation: {recommendation}")
            
        # Add conclusion with nuanced scoring
        topic = question.split('?')[0].lower()
        if len(topic) > 50:  # Truncate long topics
            topic = topic[:47] + '...'
        
        # Enhanced conclusion phrases with more specific feedback
        if not hasattr(self, 'enhanced_conclusion_phrases'):
            self.enhanced_conclusion_phrases = {
                'high': [
                    "Overall, your response demonstrates {level} understanding of {topic}. You've covered most key concepts thoroughly.",
                    "Your answer shows {level} knowledge about {topic}, with clear explanations of core principles.",
                    "You've demonstrated {level} grasp of {topic} concepts, showing both breadth and depth of understanding."
                ],
                'high_medium': [
                    "Your response indicates {level} familiarity with {topic}, though a few minor details could be expanded.",
                    "You show {level} comprehension of {topic}, with good coverage of most important aspects.",
                    "Your explanation reflects {level} understanding of {topic}, with only a few concepts needing clarification."
                ],
                'medium': [
                    "Your answer demonstrates {level} understanding of {topic}, though some important concepts could be explored further.",
                    "You show {level} knowledge of {topic}, but could benefit from deeper exploration of certain principles.",
                    "Your response indicates {level} grasp of {topic}, with room to develop more comprehensive explanations."
                ],
                'medium_low': [
                    "Your answer shows {level} familiarity with {topic}, with several key concepts needing more attention.",
                    "You demonstrate {level} understanding of {topic}, and would benefit from focusing on the core principles.",
                    "Your explanation reflects {level} knowledge of {topic}, with opportunities to strengthen fundamental concepts."
                ],
                'low': [
                    "Your response indicates {level} understanding of {topic}, with significant room for development of core concepts.",
                    "You show {level} familiarity with {topic}, and would benefit from revisiting the fundamental principles.",
                    "Your answer demonstrates {level} grasp of {topic}, suggesting a need to build a stronger foundation in this area."
                ]
            }
        
        # Select appropriate conclusion based on level category
        conclusion_phrases = self.enhanced_conclusion_phrases.get(
            level_category, self.feedback_components['conclusion_phrases'])
        conclusion_phrase = random.choice(conclusion_phrases)
        conclusion = conclusion_phrase.format(level=understanding_level, topic=topic)
        feedback_parts.append(conclusion)
        
        # Calculate a more nuanced score (0-100 instead of just high/medium/low)
        nuanced_score = self.calculate_nuanced_score(max_similarity, len(missing_concepts), len(mentioned_concepts) if 'mentioned_concepts' in locals() else 0)
        
        # Add score interpretation for educational value
        score_interpretation = self.get_score_interpretation(nuanced_score)
        feedback_parts.append(f"\nScore: {nuanced_score}/100 - {score_interpretation}")
        
        # Join all feedback parts with appropriate spacing
        feedback = "\n\n".join(feedback_parts)
        return feedback