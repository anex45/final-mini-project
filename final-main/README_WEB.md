# Python Interview System - Web Interface

This is the web interface for the Python Interview System, providing a user-friendly frontend to interact with the AI-powered interview backend.

## Features

- Interactive web-based interview experience
- Question category selection
- Real-time answer evaluation and feedback
- Score visualization
- Interview summary and export
- Responsive design for desktop and mobile devices

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- All dependencies from the original Python Interview System
- Flask and other web dependencies

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_web.txt
```

2. Ensure the backend models are trained (if not already done):

```bash
python main_unified.py
```

3. Start the web application:

```bash
python app.py
```

4. Open your browser and navigate to:

```
http://localhost:5000
```

## Usage

1. Enter your name on the welcome screen and click "Start Interview"
2. Select a question category from the available options
3. Answer the Python interview question in the text area
4. Submit your answer to receive AI-generated feedback and scoring
5. Continue with more questions or end the interview to see your summary
6. Download your interview results as a JSON file for future reference

## Architecture

The web interface consists of:

- **Frontend**: HTML, CSS, and JavaScript for the user interface
- **Backend**: Flask web server that connects to the Python Interview System
- **API Endpoints**: RESTful endpoints for retrieving questions, submitting answers, and managing the interview flow

## Integration with Existing System

The web interface integrates with the existing Python Interview System by:

1. Using the same question database and categories
2. Leveraging the trained BERT, Seq2Seq, and T5 models for answer evaluation
3. Generating the same detailed feedback as the terminal version
4. Maintaining session data in a compatible format

## Development

To modify the web interface:

- Frontend files are located in `templates/` and `static/` directories
- Backend API is defined in `app.py`
- Styling can be customized in `static/css/styles.css`

## Troubleshooting

- If the web server fails to start, ensure no other application is using port 5000
- If models fail to load, verify that you've run the training process with `main_unified.py`
- For any JavaScript errors, check the browser console for detailed information