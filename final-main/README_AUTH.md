# Implementing Clerk Authentication in Python Interview System

## Overview

This document provides instructions for implementing Clerk authentication in the Python Interview System. Clerk is a complete authentication and user management solution that provides secure sign-in, sign-up, and user profile management.

## Prerequisites

- Python 3.8 or higher
- Flask web application (existing Python Interview System)
- Clerk account (sign up at https://clerk.dev)

## Setup Instructions

### 1. Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements_clerk.txt
```

### 2. Configure Clerk

1. Sign up for a Clerk account at https://clerk.dev
2. Create a new application in the Clerk dashboard
3. Get your API keys from the Clerk dashboard:
   - Clerk API Key
   - Clerk Frontend API
   - Clerk JWT Verification Key

### 3. Configure Environment Variables

Update the `.env` file with your Clerk API keys:

```
CLERK_API_KEY=your_clerk_api_key
CLERK_FRONTEND_API=your_clerk_frontend_api
CLERK_JWT_KEY=your_clerk_jwt_verification_key
FLASK_SECRET_KEY=your_flask_secret_key
FLASK_DEBUG=True
```

### 4. Run the Application

Start the application with authentication enabled:

```bash
python app_with_auth.py
```

## Implementation Details

### Authentication Flow

1. Users visit the application and are presented with a login/signup screen
2. After successful authentication, users can access the interview system
3. All API requests include the authentication token in the Authorization header
4. Protected routes verify the token before processing requests

### Files Modified/Added

- `app_with_auth.py`: Flask application with Clerk authentication
- `auth.py`: Authentication module with Clerk integration
- `templates/index_with_auth.html`: Updated frontend with authentication components
- `static/js/app_with_auth.js`: Updated JavaScript with authentication logic
- `static/css/auth-styles.css`: Styles for authentication components
- `requirements_clerk.txt`: Updated dependencies including Clerk SDK
- `.env`: Environment variables for configuration

## Using the Authentication System

### Protected Routes

Routes that require authentication are protected with the `@login_required` decorator:

```python
@app.route('/api/protected-route')
@login_required
def protected_route():
    # Only authenticated users can access this route
    return jsonify({'message': 'This is a protected route'})
```

### Accessing User Information

User information is available in the `request.user` object for authenticated routes:

```python
@app.route('/api/user-info')
@login_required
def user_info():
    user_id = request.user.get('id')
    return jsonify({'user_id': user_id})
```

### Frontend Authentication

The frontend JavaScript handles authentication state and includes the authentication token with API requests:

```javascript
async function authenticatedFetch(url, options = {}) {
    if (!interviewState.authToken) {
        throw new Error('No authentication token available');
    }
    
    const headers = {
        'Authorization': `Bearer ${interviewState.authToken}`,
        'Content-Type': 'application/json',
        ...options.headers
    };
    
    return fetch(url, {
        ...options,
        headers
    });
}
```

## Troubleshooting

- **Authentication Errors**: Check that your Clerk API keys are correctly set in the `.env` file
- **Token Verification Failures**: Ensure the CLERK_JWT_KEY is correctly configured
- **Frontend Issues**: Check browser console for JavaScript errors related to Clerk initialization

## Security Considerations

- Never expose your Clerk API keys in client-side code
- Always use HTTPS in production environments
- Implement proper CSRF protection for form submissions
- Regularly rotate your API keys and secrets