# Python Interview System - REST API Documentation

## Overview

This document provides comprehensive documentation for the Python Interview System REST API. The API allows frontend applications to interact with the interview system, providing endpoints for retrieving question categories, fetching questions, submitting answers for evaluation, and saving interview results.

## API Base URL

The API is accessible at: `http://localhost:5000`

## Authentication

The API supports authentication through Clerk. When authentication is enabled, all API endpoints (except `/api/auth/status` and `/api/health`) require a valid authentication token.

### Authentication Headers

To access protected endpoints, include the following header in your requests:

```
Authorization: Bearer YOUR_TOKEN
```

### Checking Authentication Status

- **URL**: `/api/auth/status`
- **Method**: `GET`
- **Headers**: `Authorization: Bearer YOUR_TOKEN`
- **Response Example (Authenticated)**:
  ```json
  {
    "status": "success",
    "data": {
      "isAuthenticated": true,
      "user": {
        "id": "user_123",
        "email": "user@example.com",
        "name": "John Doe"
      }
    }
  }
  ```
- **Response Example (Not Authenticated)**:
  ```json
  {
    "status": "error",
    "message": "Invalid token",
    "data": {
      "isAuthenticated": false
    }
  }
  ```

## API Endpoints

### Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns basic API information and available endpoints
- **Authentication**: Not required
- **Response Example**:
  ```json
  {
    "status": "success",
    "message": "Python Interview System REST API",
    "version": "1.0",
    "endpoints": [
      {"path": "/api/categories", "method": "GET", "description": "Get all question categories"},
      {"path": "/api/question", "method": "GET", "description": "Get a question from a category"},
      {"path": "/api/submit", "method": "POST", "description": "Submit an answer for evaluation"},
      {"path": "/api/end", "method": "POST", "description": "End the interview and save results"}
    ]
  }
  ```

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Description**: Check if the API is running properly
- **Authentication**: Not required
- **Response Example**:
  ```json
  {
    "status": "success",
    "message": "API is healthy",
    "timestamp": "2023-06-01T12:34:56.789Z"
  }
  ```

### Get Categories

- **URL**: `/api/categories`
- **Method**: `GET`
- **Description**: Get all available question categories
- **Authentication**: Required
- **Response Example**:
  ```json
  {
    "status": "success",
    "data": [
      {"id": "python_basics", "name": "Python Basics"},
      {"id": "data_structures", "name": "Data Structures"},
      {"id": "algorithms", "name": "Algorithms"},
      {"id": "object_oriented_programming", "name": "Object Oriented Programming"}
    ]
  }
  ```

### Get Question

- **URL**: `/api/question`
- **Method**: `GET`
- **Query Parameters**: `category` (required) - The category ID to get a question from
- **Authentication**: Required
- **Response Example (Success)**:
  ```json
  {
    "status": "success",
    "data": {
      "question": "Explain the difference between a list and a tuple in Python.",
      "category": "python_basics"
    }
  }
  ```
- **Response Example (Error)**:
  ```json
  {
    "status": "error",
    "message": "Category parameter is required"
  }
  ```

### Submit Answer

- **URL**: `/api/submit`
- **Method**: `POST`
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "question": "Explain the difference between a list and a tuple in Python.",
    "answer": "Lists are mutable while tuples are immutable. Lists use square brackets and tuples use parentheses."
  }
  ```
- **Response Example (Success)**:
  ```json
  {
    "status": "success",
    "data": {
      "score": 85,
      "feedback": "Good explanation of the key difference. You could also mention that tuples are generally faster than lists and are used for heterogeneous data while lists are used for homogeneous data."
    }
  }
  ```
- **Response Example (Error)**:
  ```json
  {
    "status": "error",
    "message": "Question and answer are required"
  }
  ```

### End Interview

- **URL**: `/api/end`
- **Method**: `POST`
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "candidateName": "John Doe",
    "results": [
      {
        "question": "Explain the difference between a list and a tuple in Python.",
        "answer": "Lists are mutable while tuples are immutable.",
        "score": 85,
        "feedback": "Good explanation of the key difference."
      },
      {
        "question": "What is a decorator in Python?",
        "answer": "A decorator is a design pattern in Python that allows a user to add new functionality to an existing object without modifying its structure.",
        "score": 90,
        "feedback": "Excellent explanation of decorators."
      }
    ]
  }
  ```
- **Response Example (Success)**:
  ```json
  {
    "status": "success",
    "data": {
      "filename": "sessions/interview_20230601_123456.json",
      "timestamp": "20230601_123456"
    }
  }
  ```
- **Response Example (Error)**:
  ```json
  {
    "status": "error",
    "message": "Candidate name and results are required"
  }
  ```

## Error Handling

All API endpoints follow a consistent error handling pattern. When an error occurs, the API returns a JSON response with the following structure:

```json
{
  "status": "error",
  "message": "Description of the error",
  "error_code": 400
}
```

Common HTTP status codes:

- `200 OK`: The request was successful
- `400 Bad Request`: The request was invalid or missing required parameters
- `401 Unauthorized`: Authentication is required or the provided token is invalid
- `404 Not Found`: The requested resource was not found
- `405 Method Not Allowed`: The HTTP method is not supported for the requested endpoint
- `500 Internal Server Error`: An unexpected error occurred on the server

## Integration Example

Here's a simple example of how to integrate with the API using JavaScript:

```javascript
// Example: Fetching categories
async function getCategories() {
  const token = 'YOUR_AUTH_TOKEN'; // Get this from your authentication system
  
  try {
    const response = await fetch('http://localhost:5000/api/categories', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
      // Process the categories
      console.log(data.data);
    } else {
      // Handle error
      console.error(data.message);
    }
  } catch (error) {
    console.error('API request failed:', error);
  }
}
```

## Setup and Deployment

1. **Install Dependencies**:
   ```
   pip install -r requirements_web.txt
   ```

2. **Configure Environment Variables**:
   Create a `.env` file with the following variables:
   ```
   FLASK_SECRET_KEY=your_secret_key
   CLERK_API_KEY=your_clerk_api_key (if using authentication)
   CLERK_FRONTEND_API=your_clerk_frontend_api (if using authentication)
   PORT=5000 (optional, default is 5000)
   ```

3. **Run the API Server**:
   ```
   python app.py
   ```

4. **Access the API**:
   The API will be available at `http://localhost:5000`

## Notes

- The API uses Flask-CORS to handle cross-origin requests, so you can host your frontend on a different domain or port
- All responses are in JSON format
- Error responses include appropriate HTTP status codes and error messages
- Authentication is optional and can be disabled if not needed