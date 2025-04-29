/**
 * Python Interview System - Frontend JavaScript
 * This file handles the frontend functionality and API interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const startScreen = document.getElementById('start-screen');
    const categoryScreen = document.getElementById('category-screen');
    const questionScreen = document.getElementById('question-screen');
    const feedbackScreen = document.getElementById('feedback-screen');
    const resultsScreen = document.getElementById('results-screen');
    
    const candidateNameInput = document.getElementById('candidate-name');
    const startBtn = document.getElementById('start-btn');
    const categoriesContainer = document.getElementById('categories-container');
    const questionText = document.getElementById('question-text');
    const answerInput = document.getElementById('answer-input');
    const submitBtn = document.getElementById('submit-btn');
    const scoreValue = document.getElementById('score-value');
    const feedbackText = document.getElementById('feedback-text');
    const nextQuestionBtn = document.getElementById('next-question-btn');
    const endInterviewBtn = document.getElementById('end-interview-btn');
    const resultsSummary = document.getElementById('results-summary');
    const resultsDetails = document.getElementById('results-details');
    const restartBtn = document.getElementById('restart-btn');
    
    // Interview state
    let candidateName = '';
    let currentCategory = '';
    let currentQuestion = '';
    let interviewResults = [];
    
    // API endpoints
    const API_BASE_URL = window.location.origin;
    const API_ENDPOINTS = {
        categories: `${API_BASE_URL}/api/categories`,
        question: `${API_BASE_URL}/api/question`,
        submit: `${API_BASE_URL}/api/submit`,
        end: `${API_BASE_URL}/api/end`
    };
    
    // Event Listeners
    startBtn.addEventListener('click', startInterview);
    submitBtn.addEventListener('click', submitAnswer);
    nextQuestionBtn.addEventListener('click', getNextQuestion);
    endInterviewBtn.addEventListener('click', showResults);
    restartBtn.addEventListener('click', resetInterview);
    
    // Functions
    function startInterview() {
        candidateName = candidateNameInput.value.trim();
        if (!candidateName) {
            alert('Please enter your name to start the interview.');
            return;
        }
        
        startScreen.style.display = 'none';
        categoryScreen.style.display = 'block';
        
        // Load categories
        loadCategories();
    }
    
    function loadCategories() {
        fetch(API_ENDPOINTS.categories)
            .then(response => response.json())
            .then(categories => {
                categoriesContainer.innerHTML = '';
                
                categories.forEach(category => {
                    const categoryElement = document.createElement('button');
                    categoryElement.className = 'list-group-item list-group-item-action category-item';
                    categoryElement.textContent = category.name;
                    categoryElement.addEventListener('click', () => selectCategory(category.id));
                    categoriesContainer.appendChild(categoryElement);
                });
            })
            .catch(error => {
                console.error('Error loading categories:', error);
                alert('Failed to load question categories. Please try again.');
            });
    }
    
    function selectCategory(categoryId) {
        currentCategory = categoryId;
        categoryScreen.style.display = 'none';
        questionScreen.style.display = 'block';
        
        // Get first question from selected category
        getQuestionFromCategory(categoryId);
    }
    
    function getQuestionFromCategory(categoryId) {
        fetch(`${API_ENDPOINTS.question}?category=${categoryId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                currentQuestion = data.question;
                questionText.textContent = currentQuestion;
                answerInput.value = '';
            })
            .catch(error => {
                console.error('Error getting question:', error);
                alert('Failed to load question. Please try again.');
            });
    }
    
    function submitAnswer() {
        const answer = answerInput.value.trim();
        if (!answer) {
            alert('Please provide an answer before submitting.');
            return;
        }
        
        const requestData = {
            question: currentQuestion,
            answer: answer
        };
        
        fetch(API_ENDPOINTS.submit, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Store result
                interviewResults.push({
                    question: currentQuestion,
                    answer: answer,
                    score: data.score,
                    feedback: data.feedback
                });
                
                // Show feedback
                scoreValue.textContent = data.score;
                feedbackText.textContent = data.feedback;
                
                questionScreen.style.display = 'none';
                feedbackScreen.style.display = 'block';
            })
            .catch(error => {
                console.error('Error submitting answer:', error);
                alert('Failed to submit answer. Please try again.');
            });
    }
    
    function getNextQuestion() {
        feedbackScreen.style.display = 'none';
        categoryScreen.style.display = 'block';
    }
    
    function showResults() {
        feedbackScreen.style.display = 'none';
        resultsScreen.style.display = 'block';
        
        // Calculate average score
        const totalScore = interviewResults.reduce((sum, result) => sum + result.score, 0);
        const averageScore = interviewResults.length > 0 ? Math.round(totalScore / interviewResults.length) : 0;
        
        // Display summary
        resultsSummary.innerHTML = `
            <h3>Candidate: ${candidateName}</h3>
            <h4>Average Score: ${averageScore}%</h4>
            <p>Total Questions: ${interviewResults.length}</p>
        `;
        
        // Display detailed results
        resultsDetails.innerHTML = '';
        interviewResults.forEach(result => {
            const resultElement = document.createElement('div');
            resultElement.className = 'result-item';
            resultElement.innerHTML = `
                <div class="result-question">${result.question}</div>
                <div class="result-answer">Your Answer: ${result.answer}</div>
                <div class="result-score">Score: ${result.score}%</div>
                <div class="result-feedback">Feedback: ${result.feedback}</div>
            `;
            resultsDetails.appendChild(resultElement);
        });
        
        // Save results to server
        saveResults();
    }
    
    function saveResults() {
        const requestData = {
            candidateName: candidateName,
            results: interviewResults
        };
        
        fetch(API_ENDPOINTS.end, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                console.log('Interview results saved:', data.filename);
            })
            .catch(error => {
                console.error('Error saving results:', error);
                alert('Failed to save interview results, but you can still view them here.');
            });
    }
    
    function resetInterview() {
        // Reset state
        candidateName = '';
        currentCategory = '';
        currentQuestion = '';
        interviewResults = [];
        
        // Reset UI
        candidateNameInput.value = '';
        answerInput.value = '';
        
        // Show start screen
        resultsScreen.style.display = 'none';
        startScreen.style.display = 'block';
    }
});