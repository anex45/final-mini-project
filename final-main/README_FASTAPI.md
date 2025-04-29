# Python Interview System - FastAPI Implementation

## Overview

This document provides instructions for using the FastAPI implementation of the Python Interview System API. The FastAPI version offers the same functionality as the Flask version but with improved performance, automatic documentation, and proper CORS support.

## CORS Support

Cross-Origin Resource Sharing (CORS) has been fully enabled in this FastAPI implementation with the following configuration:

- **Allow all origins** (`allow_origins=["*"]`): The API can be accessed from any domain
- **Allow all methods** (`allow_methods=["*"]`): All HTTP methods (GET, POST, PUT, DELETE, etc.) are supported
- **Allow all headers** (`allow_headers=["*"]`): All HTTP headers are supported
- **Allow credentials** (`allow_credentials=True`): Cookies and authentication can be included in cross-origin requests

This configuration ensures that your frontend application can communicate with the API regardless of where it's hosted.

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_fastapi.txt
```

## Running the FastAPI Application

To start the FastAPI server:

```bash
python fastapi_app.py
```

Or you can use Uvicorn directly:

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 5000 --reload
```

The API will be available at http://localhost:5000/

## API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, you can access:

- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

These documentation pages allow you to explore and test all available endpoints.

## API Endpoints

The FastAPI implementation provides the same endpoints as the original Flask API:

- `GET /api/categories` - Get all question categories
- `GET /api/question` - Get a question from a category
- `POST /api/submit` - Submit an answer for evaluation
- `POST /api/end` - End the interview and save results

## Differences from Flask Implementation

- Improved request validation using Pydantic models
- Automatic API documentation with Swagger UI and ReDoc
- Better performance with ASGI server (Uvicorn)
- Proper CORS middleware configuration
- Type hints throughout the codebase

## Notes

- The FastAPI implementation can run alongside the Flask implementation without conflicts (just use different ports)
- All data is stored in the same format and location as the Flask version
- Authentication is handled the same way as in the Flask version (if available)