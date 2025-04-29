#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask Web Application for Python Interview System with Clerk Authentication

This script serves as the web interface for the Python interview system,
providing API endpoints for the frontend to interact with the backend.
It includes Clerk authentication to secure the application.
"""

import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv

# Import system components
from feedback_generator import FeedbackGenerator
from feedback_generator_seq2seq import FeedbackGeneratorSeq2Seq
from feedback_generator_t5 import FeedbackGeneratorT5
from feedback_orchestrator import FeedbackOrchestrator
from utils import ensure_directories_exist

# Import authentication module
from auth import ClerkAuth, login_required

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

# Enable CORS for the frontend
CORS(app)

# Initialize Clerk authentication
clerk_auth = ClerkAuth(app)

# Global variables
feedback_gen = None
sessions = []

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

# Authentication Routes
@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Check if the user is authenticated."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'isAuthenticated': False})
    
    token = auth_header.split(' ')[1]
    try:
        # Verify token with Clerk API
        headers = {
            'Authorization': f'Bearer {os.getenv("CLERK_API_KEY")}',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            f'https://api.clerk.dev/v1/sessions/verify',
            headers=headers,
            params={'session_token': token}
        )
        
        if response.status_code != 200:
            return jsonify({'isAuthenticated': False})
        
        user_data = response.json()
        return jsonify({
            'isAuthenticated': True,
            'user': {
                'id': user_data.get('id'),
                'email': user_data.get('email'),
                'name': user_data.get('first_name', '') + ' ' + user_data.get('last_name', '')
            }
        })
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return jsonify({'isAuthenticated': False})

# Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', clerk_frontend_api=os.getenv('CLERK_FRONTEND_API'))

@app.route('/api/categories', methods=['GET'])
@login_required
def get_categories():
    """Get all available question categories."""
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
    
    return jsonify(formatted_categories)

@app.route('/api/question', methods=['GET'])
@login_required
def get_question():
    """Get a random question from the specified category."""
    category = request.args.get('category')
    if not category:
        return jsonify({'error': 'Category parameter is required'}), 400
    
    # Load questions and filter by category
    all_questions = load_questions()
    category_questions = [q for q in all_questions if q['category'] == category]
    
    if not category_questions:
        return jsonify({'error': 'No questions found for this category'}), 404
    
    # For simplicity, return the first question in the category
    # In a real implementation, you would track which questions have been asked
    # and return a random unasked question
    return jsonify({
        'question': category_questions[0]['question'],
        'category': category
    })

@app.route('/api/submit', methods=['POST'])
@login_required
def submit_answer():
    """Submit an answer for evaluation."""
    data = request.json
    if not data or 'question' not in data or 'answer' not in data:
        return jsonify({'error': 'Question and answer are required'}), 400
    
    question = data['question']
    answer = data['answer']
    
    try:
        # Initialize feedback generator if not already done
        if feedback_gen is None:
            initialize_feedback_generator()
        
        # Get score and feedback
        score = feedback_gen.get_score(question, answer)
        feedback = feedback_gen.generate_feedback(question, answer)
        
        # Add to session data
        session_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'score': score,
            'feedback': feedback,
            'user_id': request.user.get('id') if hasattr(request, 'user') else None
        }
        sessions.append(session_entry)
        
        return jsonify({
            'score': score,
            'feedback': feedback
        })
    except Exception as e:
        print(f"Error generating feedback: {str(e)}")
        return jsonify({
            'error': 'Failed to evaluate answer',
            'score': 50,  # Default score
            'feedback': "Sorry, I couldn't generate detailed feedback for this response."
        }), 500

@app.route('/api/end', methods=['POST'])
@login_required
def end_interview():
    """End the interview and save the session."""
    data = request.json
    if not data or 'candidateName' not in data or 'results' not in data:
        return jsonify({'error': 'Candidate name and results are required'}), 400
    
    # Create directory if it doesn't exist
    if not os.path.exists('sessions'):
        os.makedirs('sessions')
    
    # Generate filename with timestamp and user ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    user_id = request.user.get('id') if hasattr(request, 'user') else 'anonymous'
    filename = f"sessions/interview_{user_id}_{timestamp}.json"
    
    # Add user ID to the data
    data['user_id'] = user_id
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return jsonify({'success': True, 'filename': filename})

# Static files route for development
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def main():
    """Main function to run the Flask application."""
    # Ensure all required directories exist
    ensure_directories_exist()
    
    # Initialize feedback generator
    initialize_feedback_generator()
    
    # Run the Flask app
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()