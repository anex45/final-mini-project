#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Authentication module for Python Interview System

This module handles authentication using Clerk's authentication service.
It provides middleware and utilities for protecting routes and verifying user sessions.
"""

import os
import json
from functools import wraps
from flask import request, jsonify, session, redirect, url_for
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clerk API configuration
CLERK_API_KEY = os.getenv('CLERK_API_KEY')
CLERK_FRONTEND_API = os.getenv('CLERK_FRONTEND_API')
CLERK_JWT_KEY = os.getenv('CLERK_JWT_KEY')

class ClerkAuth:
    """Clerk authentication handler for Flask applications."""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the Clerk authentication with a Flask app."""
        self.app = app
        app.config['CLERK_API_KEY'] = CLERK_API_KEY
        app.config['CLERK_FRONTEND_API'] = CLERK_FRONTEND_API
        app.config['CLERK_JWT_KEY'] = CLERK_JWT_KEY
        
        # Add authentication middleware
        @app.before_request
        def auth_middleware():
            # Skip authentication for public routes
            if request.path in ['/login', '/signup', '/'] or \
               request.path.startswith('/static/') or \
               request.path.startswith('/api/auth/'):
                return None
            
            # Check for authentication token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Unauthorized - No valid token provided'}), 401
            
            token = auth_header.split(' ')[1]
            try:
                # Verify token with Clerk API
                headers = {
                    'Authorization': f'Bearer {CLERK_API_KEY}',
                    'Content-Type': 'application/json'
                }
                response = requests.get(
                    f'https://api.clerk.dev/v1/sessions/verify',
                    headers=headers,
                    params={'session_token': token}
                )
                
                if response.status_code != 200:
                    return jsonify({'error': 'Unauthorized - Invalid token'}), 401
                
                # Store user info in request context
                user_data = response.json()
                request.user = user_data
                
            except Exception as e:
                print(f"Authentication error: {str(e)}")
                return jsonify({'error': 'Authentication failed'}), 401
            
            return None

# Decorator for protecting routes
def login_required(f):
    """Decorator to protect routes that require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized - No valid token provided'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            # Verify token with Clerk API
            headers = {
                'Authorization': f'Bearer {CLERK_API_KEY}',
                'Content-Type': 'application/json'
            }
            response = requests.get(
                f'https://api.clerk.dev/v1/sessions/verify',
                headers=headers,
                params={'session_token': token}
            )
            
            if response.status_code != 200:
                return jsonify({'error': 'Unauthorized - Invalid token'}), 401
            
            # Store user info in request context
            user_data = response.json()
            request.user = user_data
            
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Helper functions for authentication
def get_user_info(token):
    """Get user information from Clerk using the session token."""
    try:
        headers = {
            'Authorization': f'Bearer {CLERK_API_KEY}',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            f'https://api.clerk.dev/v1/users/me',
            headers=headers,
            params={'session_token': token}
        )
        
        if response.status_code != 200:
            return None
        
        return response.json()
    except Exception as e:
        print(f"Error getting user info: {str(e)}")
        return None