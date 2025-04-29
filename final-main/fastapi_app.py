#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI REST API for Python Interview System

This script serves as the pure REST API backend for the Python interview system,
providing RESTful endpoints for any frontend to interact with the system.
All responses are in JSON format for easy integration with any client application.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import wraps

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # For handling cross-origin requests
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import authentication module (optional)
try:
    from auth import ClerkAuth, login_required
    auth_available = True
except ImportError:
    auth_available = False
    # Define a simple login_required decorator when auth module is not available
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function

# Import system components
import sys
import os

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules from the parent directory
from feedback_generator import FeedbackGenerator
from feedback_generator_seq2seq import FeedbackGeneratorSeq2Seq
from feedback_generator_t5 import FeedbackGeneratorT5
from feedback_orchestrator import FeedbackOrchestrator
from utils import ensure_directories_exist

# Initialize FastAPI app
app = FastAPI(
    title="Python Interview System API",
    description="REST API for the Python Interview System",
    version="1.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Global variables
feedback_gen = None
sessions = []

# Pydantic models for request/response validation
class CategoryResponse(BaseModel):
    id: str
    name: str

class QuestionResponse(BaseModel):
    id: str
    question: str
    category: str

class AnswerSubmission(BaseModel):
    question_id: str
    answer: str
    session_id: Optional[str] = None

class EndInterviewRequest(BaseModel):
    session_id: str

# Load questions from training data
def load_questions(filepath='data/training_data.json'):
    """Load interview questions from training data."""
    if not os.path.exists(filepath):
        print(f"Warning: Training data file {filepath} not found.")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract unique questions with their categories
    questions = []
    seen_questions = set()
    
    for item in data:
        question = item['question']
        if question not in seen_questions:
            questions.append({
                'question': question,
                'category': item['category']
            })
            seen_questions.add(question)
    
    return questions

# Initialize feedback generator
def initialize_feedback_generator(use_seq2seq=False, use_t5=False, use_combined=False):
    """Initialize the appropriate feedback generator based on configuration."""
    global feedback_gen
    
    if use_combined and os.path.exists('models/bert_feedback_final.pt'):
        print("Using Combined Orchestrator for comprehensive feedback generation.")
        feedback_gen = FeedbackOrchestrator(use_all_models=True)
    elif use_t5 and os.path.exists('models/t5_feedback_final.pt'):
        print("Using T5 model for advanced feedback generation.")
        feedback_gen = FeedbackGeneratorT5()
    elif use_seq2seq and os.path.exists('models/bert_feedback_final.pt'):
        print("Using BERT sequence-to-sequence model for feedback generation.")
        feedback_gen = FeedbackGeneratorSeq2Seq()
    else:
        print("Using standard feedback generator.")
        feedback_gen = FeedbackGenerator()

# Routes
@app.get("/")
async def index():
    """API root endpoint providing information about available endpoints."""
    return {
        'status': 'success',
        'message': 'Python Interview System REST API',
        'version': '1.0',
        'endpoints': [
            {'path': '/api/categories', 'method': 'GET', 'description': 'Get all question categories'},
            {'path': '/api/question', 'method': 'GET', 'description': 'Get a question from a category'},
            {'path': '/api/submit', 'method': 'POST', 'description': 'Submit an answer for evaluation'},
            {'path': '/api/end', 'method': 'POST', 'description': 'End the interview and save results'}
        ]
    }

@app.get("/api")
async def api_info():
    """API information endpoint (legacy support)."""
    return {
        'status': 'success',
        'message': 'Python Interview System REST API is running',
        'endpoints': [
            {'path': '/api/categories', 'method': 'GET', 'description': 'Get all question categories'},
            {'path': '/api/question', 'method': 'GET', 'description': 'Get a question from a category'},
            {'path': '/api/submit', 'method': 'POST', 'description': 'Submit an answer for evaluation'},
            {'path': '/api/end', 'method': 'POST', 'description': 'End the interview and save results'}
        ]
    }

@app.get("/api/categories", response_model=Dict[str, Any])
async def get_categories():
    """Get all available question categories.
    
    Returns:
        JSON array of category objects with id and name properties
    """
    try:
        questions = load_questions()
        categories = list(set([q['category'] for q in questions]))
        
        # Format categories for frontend
        formatted_categories = []
        for category in categories:
            # Format category name for display (replace underscores with spaces and capitalize)
            display_name = category.replace('_', ' ').title()
            formatted_categories.append({
                'id': category,
                'name': display_name
            })
        
        return {
            'status': 'success',
            'data': formatted_categories
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': str(e)
            }
        )

@app.get("/api/question", response_model=Dict[str, Any])
async def get_question(category: str = None):
    """Get a random question from the specified category.
    
    Query Parameters:
        category (str): The category ID to get a question from
    """
    try:
        import random
        questions = load_questions()
        
        # Filter questions by category if specified
        if category:
            filtered_questions = [q for q in questions if q['category'] == category]
            if not filtered_questions:
                return JSONResponse(
                    status_code=404,
                    content={
                        'status': 'error',
                        'message': f'No questions found in category: {category}'
                    }
                )
            selected_question = random.choice(filtered_questions)
        else:
            selected_question = random.choice(questions)
        
        # Generate a unique ID for the question
        question_id = f"{selected_question['category']}_{int(time.time())}"
        
        return {
            'status': 'success',
            'data': {
                'id': question_id,
                'question': selected_question['question'],
                'category': selected_question['category']
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': str(e)
            }
        )

@app.post("/api/submit", response_model=Dict[str, Any])
async def submit_answer(submission: AnswerSubmission):
    """Submit an answer for evaluation.
    
    Request Body:
        question_id (str): The ID of the question being answered
        answer (str): The user's answer to evaluate
        session_id (str, optional): The session ID if continuing an existing session
    """
    try:
        # Initialize feedback generator if not already initialized
        global feedback_gen
        if feedback_gen is None:
            initialize_feedback_generator(use_seq2seq=True, use_t5=True, use_combined=True)
        
        # Extract question category from question_id
        question_category = submission.question_id.split('_')[0] if '_' in submission.question_id else 'unknown'
        
        # Generate feedback for the answer
        feedback = feedback_gen.generate_feedback(submission.answer, question_category)
        
        # Create or update session
        session_id = submission.session_id
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        # Find existing session or create new one
        session = next((s for s in sessions if s.get('id') == session_id), None)
        if not session:
            session = {
                'id': session_id,
                'start_time': datetime.now().isoformat(),
                'answers': []
            }
            sessions.append(session)
        
        # Add answer to session
        session['answers'].append({
            'question_id': submission.question_id,
            'answer': submission.answer,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'status': 'success',
            'data': {
                'session_id': session_id,
                'feedback': feedback
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': str(e)
            }
        )

@app.post("/api/end", response_model=Dict[str, Any])
async def end_interview(request: EndInterviewRequest):
    """End the interview and save results.
    
    Request Body:
        session_id (str): The session ID to end and save
    """
    try:
        # Find session
        session = next((s for s in sessions if s.get('id') == request.session_id), None)
        if not session:
            return JSONResponse(
                status_code=404,
                content={
                    'status': 'error',
                    'message': f'Session not found: {request.session_id}'
                }
            )
        
        # Update session end time
        session['end_time'] = datetime.now().isoformat()
        
        # Save session to file
        ensure_directories_exist(['sessions'])
        filename = f"sessions/{request.session_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2)
        
        return {
            'status': 'success',
            'data': {
                'message': 'Interview session saved successfully',
                'session_id': request.session_id,
                'filename': filename
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': str(e)
            }
        )

# Error handlers
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            'status': 'error',
            'message': 'Endpoint not found',
            'error_code': 404
        }
    )

@app.exception_handler(405)
async def method_not_allowed(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=405,
        content={
            'status': 'error',
            'message': 'Method not allowed',
            'error_code': 405
        }
    )

@app.exception_handler(500)
async def server_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={
            'status': 'error',
            'message': 'Internal server error',
            'error_code': 500
        }
    )

# Initialize the app
@app.on_event("startup")
async def startup_event():
    # Initialize feedback generator with all models enabled
    initialize_feedback_generator(use_seq2seq=True, use_t5=True, use_combined=True)
    print("FastAPI application started with CORS enabled")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=5000, reload=True)