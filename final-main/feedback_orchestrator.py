#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Orchestrator for Python Interview System

This module orchestrates multiple feedback generation models (BERT, Seq2Seq, and T5/BART)
to provide comprehensive, dynamic feedback by combining their strengths or selecting
the most appropriate model based on question complexity and response characteristics.
"""

import os
import json
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from feedback_generator import FeedbackGenerator
from feedback_generator_seq2seq import FeedbackGeneratorSeq2Seq
from feedback_generator_t5 import FeedbackGeneratorT5

class FeedbackOrchestrator:
    """Orchestrates multiple feedback generation models to provide comprehensive feedback."""
    
    def __init__(self, use_all_models=True, model_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_all_models = use_all_models
        
        # Initialize all available models
        self.models = {}
        self.model_available = {}
        
        # Initialize BERT model (always available as fallback)
        try:
            self.models['bert'] = FeedbackGenerator()
            self.model_available['bert'] = True
            print("BERT model initialized successfully")
        except Exception as e:
            print(f"Error initializing BERT model: {str(e)}")
            self.model_available['bert'] = False
        
        # Initialize Seq2Seq model if available
        if os.path.exists('models/bert_feedback_final.pt'):
            try:
                self.models['seq2seq'] = FeedbackGeneratorSeq2Seq()
                self.model_available['seq2seq'] = True
                print("Seq2Seq model initialized successfully")
            except Exception as e:
                print(f"Error initializing Seq2Seq model: {str(e)}")
                self.model_available['seq2seq'] = False
        else:
            self.model_available['seq2seq'] = False
        
        # Initialize T5 model if available
        if os.path.exists('models/t5_feedback_final.pt'):
            try:
                self.models['t5'] = FeedbackGeneratorT5()
                self.model_available['t5'] = True
                print("T5 model initialized successfully")
            except Exception as e:
                print(f"Error initializing T5 model: {str(e)}")
                self.model_available['t5'] = False
        else:
            self.model_available['t5'] = False
        
        # Set model weights for ensemble (default or user-provided)
        self.model_weights = model_weights or {
            'bert': 0.3,
            'seq2seq': 0.3,
            't5': 0.4
        }
        
        # Normalize weights based on available models
        self._normalize_weights()
        
        # Question complexity analyzer
        self.complexity_keywords = {
            'high': ['explain', 'compare', 'analyze', 'evaluate', 'design', 'implement', 'optimize', 'debug'],
            'medium': ['describe', 'identify', 'discuss', 'outline', 'demonstrate', 'apply', 'use'],
            'low': ['define', 'list', 'name', 'what', 'when', 'who']
        }
    
    def _normalize_weights(self):
        """Normalize model weights based on available models."""
        available_models = [model for model, available in self.model_available.items() if available]
        
        if not available_models:
            # Fallback to BERT if no models are available
            self.model_weights = {'bert': 1.0}
            self.model_available['bert'] = True
            return
        
        # Filter weights to only include available models
        filtered_weights = {model: weight for model, weight in self.model_weights.items() 
                          if model in available_models}
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(filtered_weights.values())
        if weight_sum > 0:
            self.model_weights = {model: weight/weight_sum for model, weight in filtered_weights.items()}
        else:
            # Equal weights if all weights were 0
            equal_weight = 1.0 / len(available_models)
            self.model_weights = {model: equal_weight for model in available_models}
    
    def analyze_question_complexity(self, question: str) -> str:
        """Analyze the complexity of a question to determine the most appropriate model."""
        question_lower = question.lower()
        
        # Check for high complexity indicators
        for keyword in self.complexity_keywords['high']:
            if keyword in question_lower:
                return 'high'
        
        # Check for medium complexity indicators
        for keyword in self.complexity_keywords['medium']:
            if keyword in question_lower:
                return 'medium'
        
        # Default to low complexity
        return 'low'
    
    def select_best_model(self, question: str, response: str) -> str:
        """Select the most appropriate model based on question complexity and response characteristics."""
        # Analyze question complexity
        complexity = self.analyze_question_complexity(question)
        
        # For high complexity questions, prefer T5 if available
        if complexity == 'high' and self.model_available.get('t5', False):
            return 't5'
        # For medium complexity, prefer Seq2Seq if available
        elif complexity == 'medium' and self.model_available.get('seq2seq', False):
            return 'seq2seq'
        # For low complexity or fallback, use BERT
        else:
            return 'bert'
    
    def get_score(self, question: str, response: str) -> int:
        """Get a combined score from all available models or the best model."""
        if not self.use_all_models:
            # Use only the best model for scoring
            best_model = self.select_best_model(question, response)
            return self.models[best_model].get_score(question, response)
        
        # Get scores from all available models
        scores = {}
        for model_name, model in self.models.items():
            if self.model_available.get(model_name, False):
                try:
                    scores[model_name] = model.get_score(question, response)
                except Exception as e:
                    print(f"Error getting score from {model_name} model: {str(e)}")
        
        if not scores:
            return 50  # Default score if all models fail
        
        # Calculate weighted average score
        weighted_score = 0
        total_weight = 0
        
        for model_name, score in scores.items():
            weight = self.model_weights.get(model_name, 0)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return round(weighted_score / total_weight)
        else:
            return round(sum(scores.values()) / len(scores))  # Simple average if weights sum to 0
    
    def generate_feedback(self, question: str, response: str) -> str:
        """Generate comprehensive feedback by combining outputs from multiple models or using the best model."""
        if not self.use_all_models:
            # Use only the best model for feedback
            best_model = self.select_best_model(question, response)
            return self.models[best_model].generate_feedback(question, response)
        
        # Get feedback from all available models
        feedbacks = {}
        for model_name, model in self.models.items():
            if self.model_available.get(model_name, False):
                try:
                    feedbacks[model_name] = model.generate_feedback(question, response)
                except Exception as e:
                    print(f"Error generating feedback from {model_name} model: {str(e)}")
        
        if not feedbacks:
            return "I couldn't generate detailed feedback for this response. Please try again."
        
        # If we only have one model's feedback, return it directly
        if len(feedbacks) == 1:
            return list(feedbacks.values())[0]
        
        # Combine feedback from multiple models
        return self._combine_feedback(question, response, feedbacks)
    
    def _combine_feedback(self, question: str, response: str, feedbacks: Dict[str, str]) -> str:
        """Combine feedback from multiple models using weighted ensemble strategy.

        Combination Approach:
        1. Extract key feedback points from each model
        2. Weight feedback based on model confidence and complexity
        3. Merge similar points and remove redundancy
        4. Structure feedback in a clear, organized format
        """
        # Initialize combined feedback components
        feedback_components = {
            'strengths': set(),
            'improvements': set(),
            'suggestions': set()
        }
        
        # Extract and categorize feedback points from each model
        for model_name, feedback in feedbacks.items():
            weight = self.model_weights.get(model_name, 0)
            
            # Skip if weight is 0
            if weight == 0:
                continue
            
            # Split feedback into sentences
            sentences = feedback.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Categorize feedback points
                if any(pos in sentence.lower() for pos in ['good', 'great', 'excellent', 'correct']):
                    feedback_components['strengths'].add(sentence)
                elif any(neg in sentence.lower() for neg in ['improve', 'consider', 'should', 'could']):
                    feedback_components['improvements'].add(sentence)
                elif any(sug in sentence.lower() for sug in ['try', 'suggest', 'recommend']):
                    feedback_components['suggestions'].add(sentence)
                else:
                    # Add to most appropriate category based on content
                    if 'not' in sentence.lower() or 'missing' in sentence.lower():
                        feedback_components['improvements'].add(sentence)
                    else:
                        feedback_components['strengths'].add(sentence)
        
        # Combine feedback components into a structured response
        combined_feedback = []
        
        # Add strengths section if there are any
        if feedback_components['strengths']:
            combined_feedback.append("Strengths:")
            combined_feedback.extend([f"- {point}" for point in feedback_components['strengths']])
        
        # Add improvements section if there are any
        if feedback_components['improvements']:
            if combined_feedback:  # Add spacing if there's previous content
                combined_feedback.append("")
            combined_feedback.append("Areas for Improvement:")
            combined_feedback.extend([f"- {point}" for point in feedback_components['improvements']])
        
        # Add suggestions section if there are any
        if feedback_components['suggestions']:
            if combined_feedback:  # Add spacing if there's previous content
                combined_feedback.append("")
            combined_feedback.append("Suggestions:")
            combined_feedback.extend([f"- {point}" for point in feedback_components['suggestions']])
        
        # If no feedback components were found, return a default message
        if not combined_feedback:
            return "Your response has been evaluated. While the basic concepts are present, try to provide more detailed explanations and examples in your answers."
        
        # Join all feedback sections with newlines
        return '\n'.join(combined_feedback)

    def combine_model_feedbacks(self, feedbacks: Dict[str, str]) -> Dict[str, List[str]]:
        """
        1. Extract key feedback points from each model
        2. Weight feedback based on model confidence and complexity
        3. Merge similar points and remove redundancy
        4. Structure feedback in a clear, organized format
        """
        # Initialize combined feedback components
        feedback_components = {
            'strengths': set(),
            'improvements': set(),
            'suggestions': set()
        }
        
        # Extract and categorize feedback points from each model
        for model_name, feedback in feedbacks.items():
            weight = self.model_weights.get(model_name, 0)
            
            # Skip if weight is 0
            if weight == 0:
                continue
            
            # Split feedback into sentences
            sentences = feedback.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Categorize feedback points
                if any(pos in sentence.lower() for pos in ['good', 'great', 'excellent', 'correct']):
                    feedback_components['strengths'].add(sentence)
                elif any(neg in sentence.lower() for neg in ['improve', 'consider', 'should', 'could']):
                    feedback_components['improvements'].add(sentence)
                elif any(sug in sentence.lower() for sug in ['try', 'suggest', 'recommend']):
                    feedback_components['suggestions'].add(sentence)
                else:
                    # Add to most appropriate category based on content
                    if 'not' in sentence.lower() or 'missing' in sentence.lower():
                        feedback_components['improvements'].add(sentence)
                    else:
                        feedback_components['strengths'].add(sentence)
        
        # Combine feedback components into a structured response
        combined_feedback = []
        
        # Add strengths section if there are any
        if feedback_components['strengths']:
            combined_feedback.append("Strengths:")
            combined_feedback.extend([f"- {point}" for point in feedback_components['strengths']])
        
        # Add improvements section if there are any
        if feedback_components['improvements']:
            if combined_feedback:  # Add spacing if there's previous content
                combined_feedback.append("")
            combined_feedback.append("Areas for Improvement:")
            combined_feedback.extend([f"- {point}" for point in feedback_components['improvements']])
        
        # Add suggestions section if there are any
        if feedback_components['suggestions']:
            if combined_feedback:  # Add spacing if there's previous content
                combined_feedback.append("")
            combined_feedback.append("Suggestions:")
            combined_feedback.extend([f"- {point}" for point in feedback_components['suggestions']])
        
        # If no feedback components were found, return a default message
        if not combined_feedback:
            return "Your response has been evaluated. While the basic concepts are present, try to provide more detailed explanations and examples in your answers."
        
        # Join all feedback sections with newlines
        return '\n'.join(combined_feedback)

    def _combine_feedback_advanced(self, feedbacks: Dict[str, str]) -> str:
        """
        1. Use T5 for technical accuracy
        2. Use Seq2Seq for fluency
        3. Use BERT for keyword matching
        4. Ensure diversity through similarity checks
        """
        # Extract key components from each feedback
        feedback_parts = {
            'bert': self._extract_key_points(feedbacks.get('bert', '')),
            'seq2seq': self._split_into_sentences(feedbacks.get('seq2seq', '')),
            't5': self._split_into_sentences(feedbacks.get('t5', ''))
        }

        # Create weighted selection pool
        selection_pool = []
        for model, weight in self.model_weights.items():
            if model in feedback_parts and feedback_parts[model]:
                selection_pool.extend([(p, weight) for p in feedback_parts[model]])

        # Select top elements based on weights and diversity
        combined = []
        seen_concepts = set()
        for _ in range(5):  # Target 5 key points
            if not selection_pool:
                break
            
            # Weighted random selection
            total_weight = sum(w for _, w in selection_pool)
            r = random.uniform(0, total_weight)
            upto = 0
            for i, (point, weight) in enumerate(selection_pool):
                upto += weight
                if upto >= r:
                    selected = selection_pool.pop(i)
                    break

            # Check for redundancy before adding
            concept = self._extract_main_concept(selected[0])
            if concept not in seen_concepts:
                combined.append(selected[0])
                seen_concepts.add(concept)

        # Ensure natural flow between points
        return self._structure_feedback(combined)

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key technical points using BERT's strength in keyword detection."""
        return [s for s in text.split('. ') if any(kw in s.lower() for kw in self.complexity_keywords['high'])]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split feedback into coherent sentences."""
        return [s.strip() for s in text.split('. ') if s.strip()]

    def _extract_main_concept(self, sentence: str) -> str:
        """Extract primary technical concept from a sentence."""
        return next((kw for kw in self.complexity_keywords['high'] if kw in sentence.lower()), '')

    def _structure_feedback(self, points: List[str]) -> str:
        """Structure points into coherent feedback with logical flow."""
        if not points:
            return "The response addresses the question adequately but could benefit from more detail."

        structured = [
            "Here's a comprehensive analysis of your response:",
            "1. " + points[0].capitalize()
        ]

        for i, point in enumerate(points[1:], 2):
            structured.append(f"{i}. {point}")

        structured.append("Overall, this shows good understanding but could be improved in these areas.")
        return ' '.join(structured)