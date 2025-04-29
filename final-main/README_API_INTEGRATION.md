# Python Interview System - API Integration Guide

## Overview

This document provides instructions for integrating your custom frontend with the Python Interview System API. The frontend components have been removed from the project, leaving only the API backend that provides all the necessary endpoints for interview functionality.

## API Endpoints

The following RESTful endpoints are available for frontend integration:

### 1. Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns basic API information and available endpoints
- **Response Example**:
  ```json
  {
    "status": "success",
    "message": "Python Interview System API is running",
    "endpoints": [
      "/api/categories - Get all question categories",
      "/api/question - Get a question from a category",
      "/api/submit - Submit an answer for evaluation",
      "/api/end - End the interview and save results"
    ]
  }
  ```

### 2. Get Categories

- **URL**: `/api/categories`
- **Method**: `GET`
- **Description**: Returns all available question categories
- **Response Example**:
  ```json
  [
    {
      "id": "basics",
      "name": "Python Basics"
    },
    {
      "id": "data_structures",
      "name": "Data Structures"
    }
  ]
  ```

### 3. Get Question

- **URL**: `/api/question`
- **Method**: `GET`
- **Parameters**: `category` (query parameter)
- **Description**: Returns a question from the specified category
- **Response Example**:
  ```json
  {
    "question": "Explain the difference between lists and tuples in Python.",
    "category": "data_structures"
  }
  ```

### 4. Submit Answer

- **URL**: `/api/submit`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "question": "Explain the difference between lists and tuples in Python.",
    "answer": "Lists are mutable while tuples are immutable..."
  }
  ```
- **Description**: Submits an answer for evaluation
- **Response Example**:
  ```json
  {
    "score": 85,
    "feedback": "Good explanation of mutability differences..."
  }
  ```

### 5. End Interview

- **URL**: `/api/end`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "candidateName": "John Doe",
    "results": [
      {
        "question": "Explain the difference between lists and tuples in Python.",
        "answer": "Lists are mutable while tuples are immutable...",
        "score": 85,
        "feedback": "Good explanation of mutability differences..."
      }
    ]
  }
  ```
- **Description**: Ends the interview and saves the session data
- **Response Example**:
  ```json
  {
    "success": true,
    "filename": "sessions/interview_20230615_123456.json"
  }
  ```

## Integration Steps

1. **Setup the API Backend**:
   - Install the required dependencies: `pip install -r requirements_web.txt`
   - Run the API server: `python app.py`
   - The API will be available at `http://localhost:5000`

2. **Connect Your Frontend**:
   - Configure your frontend to make API calls to the endpoints listed above
   - Ensure your frontend handles CORS properly (the backend has CORS enabled)
   - Use standard HTTP methods (GET, POST) to interact with the API

3. **Implement the Interview Flow**:
   - Start by fetching categories
   - Allow the user to select a category
   - Fetch a question from the selected category
   - Submit the user's answer for evaluation
   - Display the score and feedback
   - Repeat for additional questions
   - End the interview and save the results

## Example Frontend Technologies

You can use any frontend technology to integrate with this API, such as:

- React.js
- Vue.js
- Angular
- Plain HTML/CSS/JavaScript
- Mobile applications (React Native, Flutter, etc.)

## Notes

- The API uses Flask-CORS to handle cross-origin requests, so you can host your frontend on a different domain or port
- All responses are in JSON format
- Error responses include appropriate HTTP status codes and error messages