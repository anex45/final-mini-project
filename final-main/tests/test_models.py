import pytest
import torch
import json
import os
from model_trainer import load_training_data, train_model, save_model, load_model
from feedback_generator_seq2seq import FeedbackGeneratorSeq2Seq

@pytest.fixture
def sample_training_data():
    return [
        {
            'question': 'What is inheritance?',
            'correct_answer': 'Inheritance allows classes to inherit attributes/methods from parent classes',
            'incorrect_answers': ['Inheritance is when objects share memory', 'Inheritance means copying code']
        }
    ]

@pytest.fixture
def sample_feedback_data():
    return [
        {
            'question': 'Explain polymorphism',
            'correct_answer': 'Polymorphism allows objects of different types to be treated as common superclass',
            'detailed_feedback': 'Polymorphism enables flexible code through method overriding and interfaces'
        }
    ]

def test_data_loading(sample_training_data, tmpdir):
    # Test training data loading
    data_file = tmpdir.join('training_data.json')
    with open(data_file, 'w') as f:
        json.dump(sample_training_data, f)
    
    questions, answers, labels = load_training_data(data_file)
    assert len(questions) == 3  # 1 correct + 2 incorrect
    assert sum(labels) == 1

def test_model_training(sample_training_data, tmpdir):
    # Test end-to-end model training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    questions, answers, labels = load_training_data()
    train_questions, val_questions, train_answers, val_answers, train_labels, val_labels = \
        train_test_split(questions, answers, labels, test_size=0.1)
    
    train_dataset = InterviewDataset(train_questions, train_answers, train_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Basic training sanity check
    trained_model = train_model(model, train_dataloader, None, device, epochs=1)
    assert trained_model is not None
    
    # Test model saving/loading
    model_path = os.path.join(tmpdir, 'test_model.pt')
    save_model(trained_model, model_path)
    assert os.path.exists(model_path)
    
    loaded_model = load_model(model_path, device)
    assert loaded_model is not None

def test_feedback_generation(sample_feedback_data):
    # Test feedback generation pipeline
    generator = FeedbackGeneratorSeq2Seq()
    
    # Test with sample question/answer
    feedback = generator.generate_feedback(
        'What is polymorphism?',
        'Polymorphism is when objects change shape'
    )
    
    assert isinstance(feedback, str)
    assert len(feedback) > 100
    assert 'polymorphism' in feedback.lower()
    
    # Test classification integration
    assert 'Classification:' in feedback
    assert 'Confidence:' in feedback

def test_error_handling():
    # Test invalid data handling
    with pytest.raises(FileNotFoundError):
        load_training_data('non_existent.json')
    
    # Test empty input handling
    generator = FeedbackGeneratorSeq2Seq()
    feedback = generator.generate_feedback('', '')
    assert 'Invalid input' in feedback

# Add integration test for full system workflow
def test_full_system_workflow(tmpdir):
    # Train minimal model
    device = torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Create minimal training data
    train_data = [
        {
            'question': 'Test question',
            'correct_answer': 'Correct answer',
            'incorrect_answers': ['Wrong answer']
        }
    ]
    data_file = tmpdir.join('train.json')
    with open(data_file, 'w') as f:
        json.dump(train_data, f)
    
    # Train and save model
    questions, answers, labels = load_training_data(data_file)
    train_dataset = InterviewDataset(questions, answers, labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    trained_model = train_model(model, train_dataloader, None, device, epochs=1)
    model_path = os.path.join(tmpdir, 'test_model.pt')
    save_model(trained_model, model_path)
    
    # Load and generate feedback
    generator = FeedbackGeneratorSeq2Seq(model_path=model_path)
    feedback = generator.generate_feedback('Test question', 'Wrong answer')
    
    assert 'Confidence' in feedback
    assert 'Classification: Incorrect' in feedback